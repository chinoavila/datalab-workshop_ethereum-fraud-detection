import requests
import time
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# === CONFIGURACIÓN ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
api_key = os.getenv("ALCHEMY_API_KEY")
if not api_key:
    raise ValueError("⚠️ API_KEY no está definida en las variables de entorno")
ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"

# === FUNCIONES BÁSICAS ===

@lru_cache(maxsize=None)
def is_contract_address(address):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getCode",
        "params": [address, "latest"]
    }
    try:
        response = requests.post(ALCHEMY_URL, json=payload)
        result = response.json().get("result")
        return result not in ("0x", "0x0", None)
    except Exception:
        return False

def get_eth_balance(address):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBalance",
        "params": [address, "latest"]
    }
    response = requests.post(ALCHEMY_URL, json=payload)
    result = response.json().get("result")
    return int(result, 16) / 1e18 if result else 0

def get_recent_transfers(minutes=1, max_tx=10):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{
            "fromBlock": "0x0",
            "toBlock": "latest",
            "category": ["external", "erc20"],
            "withMetadata": True,
            "excludeZeroValue": True,
            "maxCount": hex(max_tx),
            "order": "desc"
        }]
    }
    response = requests.post(ALCHEMY_URL, json=payload)
    data = response.json()
    return data.get("result", {}).get("transfers", [])

def get_historical_transfers_for_address(address, max_tx=100):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{
            "fromBlock": "0x0",
            "toBlock": "latest",
            "fromAddress": address,
            "category": ["external", "erc20"],
            "excludeZeroValue": True,
            "maxCount": hex(max_tx),
            "withMetadata": True,
            "order": "desc"
        }]
    }
    response = requests.post(ALCHEMY_URL, json=payload)
    data = response.json()
    return data.get("result", {}).get("transfers", [])

# === PROCESAMIENTO DE FEATURES ===

def extract_features(transfers):
    accounts = defaultdict(lambda: {
        "sent_values": [], "received_values": [],
        "sent_addresses": [], "received_addresses": [],
        "timestamps_sent": [], "timestamps_received": [],
        "erc20_sent_values": [], "erc20_received_values": [],
        "erc20_sent_tokens": [], "erc20_received_tokens": [],
        "erc20_sent_to": [], "erc20_received_from": [],
        "erc20_sent_to_contracts": [],
        "contract_sent_values": [],
        "created_contracts": 0
    })

    for tx in transfers:
        from_addr = tx.get("from")
        to_addr = tx.get("to")
        value = float(tx.get("value") or 0)
        timestamp = tx.get("metadata", {}).get("blockTimestamp")
        category = tx.get("category")
        asset = tx.get("asset")

        if from_addr:
            acc = accounts[from_addr]
            acc["sent_values"].append(value)
            acc["sent_addresses"].append(to_addr)
            acc["timestamps_sent"].append(timestamp)

            if not to_addr:
                acc["created_contracts"] += 1
            elif is_contract_address(to_addr):
                acc["contract_sent_values"].append(value)

            if category == "erc20":
                acc["erc20_sent_values"].append(value)
                acc["erc20_sent_tokens"].append(asset)
                acc["erc20_sent_to"].append(to_addr)
                if to_addr and is_contract_address(to_addr):
                    acc["erc20_sent_to_contracts"].append(to_addr)

        if to_addr:
            acc = accounts[to_addr]
            acc["received_values"].append(value)
            acc["received_addresses"].append(from_addr)
            acc["timestamps_received"].append(timestamp)

            if category == "erc20":
                acc["erc20_received_values"].append(value)
                acc["erc20_received_tokens"].append(asset)
                acc["erc20_received_from"].append(from_addr)

    def avg_time_diff(times):
        if len(times) < 2:
            return 0
        times = sorted(pd.to_datetime(times))
        diffs = [(t2 - t1).total_seconds() / 60 for t1, t2 in zip(times[:-1], times[1:])]
        return sum(diffs) / len(diffs) if diffs else 0

    features = []
    for addr, data in accounts.items():
        sent_times = [ts for ts in data["timestamps_sent"] if ts]
        rec_times = [ts for ts in data["timestamps_received"] if ts]
        all_times = pd.to_datetime(sent_times + rec_times)

        features.append({
            "Address": addr,
            "Avg_min_between_sent_tnx": avg_time_diff(sent_times),
            "Avg_min_between_received_tnx": avg_time_diff(rec_times),
            "Time_Diff_between_first_and_last(Mins)": (all_times.max() - all_times.min()).total_seconds() / 60 if len(all_times) > 1 else 0,
            "Sent_tnx": len(data["sent_values"]),
            "Received_tnx": len(data["received_values"]),
            "Number_of_Created_Contracts": data["created_contracts"],
            "Unique_Received_From_Addresses": len(set(data["received_addresses"])),
            "Unique_Sent_To_Addresses": len(set(data["sent_addresses"])),
            "Min_Value_Received": min(data["received_values"], default=0),
            "Max_Value_Received": max(data["received_values"], default=0),
            "Avg_Value_Received": sum(data["received_values"]) / len(data["received_values"]) if data["received_values"] else 0,
            "Min_Val_Sent": min(data["sent_values"], default=0),
            "Max_Val_Sent": max(data["sent_values"], default=0),
            "Avg_Val_Sent": sum(data["sent_values"]) / len(data["sent_values"]) if data["sent_values"] else 0,
            "Min_Value_Sent_To_Contract": min(data["contract_sent_values"], default=0),
            "Avg_Value_Sent_To_Contract": sum(data["contract_sent_values"]) / len(data["contract_sent_values"]) if data["contract_sent_values"] else 0,
            "Total_Transactions": len(data["sent_values"]) + len(data["received_values"]) + data["created_contracts"],
            "Total_Ether_Sent": sum(data["sent_values"]),
            "Total_Ether_Received": sum(data["received_values"]),
            "Total_Ether_Sent_Contracts": sum(data["contract_sent_values"]),
            "Total_Ether_Balance": get_eth_balance(addr),
            "ERC20_Total_Tnxs": len(data["erc20_sent_values"]) + len(data["erc20_received_values"]),
            "ERC20_Total_Ether_Sent_Contract": sum(data["erc20_sent_values"]),
            "ERC20_Uniq_Sent_Addr": len(set(data["erc20_sent_to"])),
            "ERC20_Uniq_Rec_Addr": len(set(data["erc20_received_from"])),
            "ERC20_Uniq_Rec_Contract_Addr": len(set(data["erc20_sent_to_contracts"])),
            "ERC20_Min_Val_Rec": min(data["erc20_received_values"], default=0),
            "ERC20_Max_Val_Rec": max(data["erc20_received_values"], default=0),
            "ERC20_Avg_Val_Rec": sum(data["erc20_received_values"]) / len(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20_Avg_Val_Sent": sum(data["erc20_sent_values"]) / len(data["erc20_sent_values"]) if data["erc20_sent_values"] else 0,
            "ERC20_Uniq_Sent_Token_Name": len(set(data["erc20_sent_tokens"]))
        })

    return pd.DataFrame(features)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    start_total = time.time()
    print("▶ Descargando transacciones recientes...")
    recent_tx = get_recent_transfers(minutes=1)
    print(f"✅ Se obtuvieron {len(recent_tx)} transacciones.")

    addresses = {tx["from"] for tx in recent_tx if tx.get("from")} | {tx["to"] for tx in recent_tx if tx.get("to")}
    print(f"📌 {len(addresses)} direcciones encontradas para analizar.\n")

    all_hist_txs = []
    print("⏳ Consultando transacciones históricas en paralelo...")

    def fetch_historical(addr):
        try:
            txs = get_historical_transfers_for_address(addr, max_tx=100)
            print(f"📥 {addr[:6]}... → {len(txs)} txs")
            return txs
        except Exception as e:
            print(f"❌ Error al procesar {addr[:6]}...: {e}")
            return []

    start_fetch = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_historical, addr): addr for addr in addresses}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_hist_txs.extend(result)
            if (i + 1) % 5 == 0 or (i + 1) == len(futures):
                elapsed = time.time() - start_fetch
                print(f"🔄 Progreso: {i+1}/{len(futures)} direcciones ({elapsed:.1f}s)")

    print(f"\n✅ Se recopilaron {len(all_hist_txs)} transacciones históricas.")
    print(f"🕒 Tiempo total de descarga: {time.time() - start_fetch:.2f} s\n")

    print("🧠 Extrayendo features...")
    start_feat = time.time()
    df_features = extract_features(all_hist_txs)
    print(f"✅ Features extraídas en {time.time() - start_feat:.2f} s")

    output_file = "historical_features_eth.csv"
    df_features.to_csv(output_file, index=False)
    print(f"📁 Archivo guardado en '{output_file}'")
    print(f"⏱️ Tiempo total del script: {time.time() - start_total:.2f} s")
