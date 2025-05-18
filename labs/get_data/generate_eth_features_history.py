import requests
import time
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import os

# === CONFIGURACIÓN ===
# Cargar variables de entorno desde el archivo .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
api_key = os.getenv("ALCHEMY_API_KEY")
if not api_key:
    raise ValueError("⚠️ API_KEY no está definida en las variables de entorno")
api_key = os.getenv("ALCHEMY_API_KEY")
ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"

# === FUNCIONES ===

def is_contract_address(address):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getCode",
        "params": [address, "latest"]
    }
    response = requests.post(ALCHEMY_URL, json=payload)
    result = response.json().get("result")
    return result not in ("0x", "0x0", None)

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
    now = int(time.time())
    start_time = now - minutes * 60

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
    x=0
    for tx in transfers:
        print(x)
        x=x+1
        from_addr = tx.get("from")
        to_addr = tx.get("to")
        value = float(tx.get("value") or 0)
        timestamp = tx.get("metadata", {}).get("blockTimestamp")
        category = tx.get("category")
        asset = tx.get("asset")

        if from_addr:
            accounts[from_addr]["sent_values"].append(value)
            accounts[from_addr]["sent_addresses"].append(to_addr)
            accounts[from_addr]["timestamps_sent"].append(timestamp)

            if not to_addr:
                accounts[from_addr]["created_contracts"] += 1

            if to_addr and is_contract_address(to_addr):
                accounts[from_addr]["contract_sent_values"].append(value)

            if category == "erc20":
                accounts[from_addr]["erc20_sent_values"].append(value)
                accounts[from_addr]["erc20_sent_tokens"].append(asset)
                accounts[from_addr]["erc20_sent_to"].append(to_addr)
                if to_addr and is_contract_address(to_addr):
                    accounts[from_addr]["erc20_sent_to_contracts"].append(to_addr)

        if to_addr:
            accounts[to_addr]["received_values"].append(value)
            accounts[to_addr]["received_addresses"].append(from_addr)
            accounts[to_addr]["timestamps_received"].append(timestamp)

            if category == "erc20":
                accounts[to_addr]["erc20_received_values"].append(value)
                accounts[to_addr]["erc20_received_tokens"].append(asset)
                accounts[to_addr]["erc20_received_from"].append(from_addr)

    features = []
    for addr, data in accounts.items():
        sent_times = [pd.to_datetime(ts) for ts in data["timestamps_sent"] if ts]
        rec_times = [pd.to_datetime(ts) for ts in data["timestamps_received"] if ts]

        def avg_time_diff(times):
            if len(times) < 2:
                return 0
            times = sorted(times)
            diffs = [(t2 - t1).total_seconds() / 60 for t1, t2 in zip(times[:-1], times[1:])]
            return sum(diffs) / len(diffs) if diffs else 0

        features.append({
            "Address": addr,
            "Avg_min_between_sent_tnx": avg_time_diff(sent_times),
            "Avg_min_between_received_tnx": avg_time_diff(rec_times),
            "Time_Diff_between_first_and_last(Mins)": (max(sent_times + rec_times) - min(sent_times + rec_times)).total_seconds() / 60 if sent_times + rec_times else 0,
            "Sent_tnx": len(data["sent_values"]),
            "Received_tnx": len(data["received_values"]),
            "Number_of_Created_Contracts": data["created_contracts"],
            "Unique_Received_From_Addresses": len(set(data["received_addresses"])),
            "Unique_Sent_To_Addresses": len(set(data["sent_addresses"])),
            "Min_Value_Received": min(data["received_values"]) if data["received_values"] else 0,
            "Max_Value_Received": max(data["received_values"]) if data["received_values"] else 0,
            "Avg_Value_Received": sum(data["received_values"]) / len(data["received_values"]) if data["received_values"] else 0,
            "Min_Val_Sent": min(data["sent_values"]) if data["sent_values"] else 0,
            "Max_Val_Sent": max(data["sent_values"]) if data["sent_values"] else 0,
            "Avg_Val_Sent": sum(data["sent_values"]) / len(data["sent_values"]) if data["sent_values"] else 0,
            "Min_Value_Sent_To_Contract": min(data["contract_sent_values"]) if data["contract_sent_values"] else 0,
            "Avg_Value_Sent_To_Contract": sum(data["contract_sent_values"]) / len(data["contract_sent_values"]) if data["contract_sent_values"] else 0,
            "Total_Transactions": len(data["sent_values"]) + len(data["received_values"]) + data["created_contracts"],
            "Total_Ether_Sent": sum(data["sent_values"]),
            "Total_Ether_Received": sum(data["received_values"]),
            "Total_Ether_Sent_Contracts": sum(data["contract_sent_values"]),
            "Total_Ether_Balance": get_eth_balance(addr),
            "ERC20_Total_Tnxs": len(data["erc20_sent_values"]) + len(data["erc20_received_values"]),
            "ERC20_Total_Ether_Sent_Contract": sum(data["erc20_sent_values"]),  # could be filtered if needed
            "ERC20_Uniq_Sent_Addr": len(set(data["erc20_sent_to"])),
            "ERC20_Uniq_Rec_Addr": len(set(data["erc20_received_from"])),
            "ERC20_Uniq_Sent_Addr.1": len(set(data["erc20_sent_to"])),  # same as above
            "ERC20_Uniq_Rec_Contract_Addr": len(set(data["erc20_sent_to_contracts"])),
            "ERC20_Min_Val_Rec": min(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20_Max_Val_Rec": max(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20_Avg_Val_Rec": sum(data["erc20_received_values"]) / len(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20_Avg_Val_Sent": sum(data["erc20_sent_values"]) / len(data["erc20_sent_values"]) if data["erc20_sent_values"] else 0,
            "ERC20_Uniq_Sent_Token_Name": len(set(data["erc20_sent_tokens"]))
        })

    return pd.DataFrame(features)


# === EJECUCIÓN ===
if __name__ == "__main__":
    print("▶ Descargando transacciones recientes...")
    recent_tx = get_recent_transfers(minutes=1)
    print(f"Se obtuvieron {len(recent_tx)} transacciones.")

    addresses = set()
    for tx in recent_tx:
        if tx.get("from"):
            addresses.add(tx["from"])
        if tx.get("to"):
            addresses.add(tx["to"])
    print(f"🧾 {len(addresses)} addresses encontradas.")

    all_hist_txs = []
    for i, addr in enumerate(addresses):
        print(f"[{i+1}/{len(addresses)}] Consultando histórico para {addr}")
        hist_txs = get_historical_transfers_for_address(addr, max_tx=100)
        all_hist_txs.extend(hist_txs)
        time.sleep(0.25)  # Evitar rate limiting

    print(f"🧠 Se recopilaron {len(all_hist_txs)} transacciones históricas.")

    df_features = extract_features(all_hist_txs)
    df_features.to_csv("historical_features_eth.csv", index=False)
    print("✅ Features guardadas en 'historical_features_eth.csv'")
