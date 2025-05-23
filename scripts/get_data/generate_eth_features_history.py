import requests
import time
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import argparse
from datetime import datetime

# === CONFIGURACIÓN ===
# Obtener la ruta absoluta al directorio raíz del proyecto
project_root = os.path.abspath(os.path.dirname(__file__))
if sys.argv[0] == "streamlit_app.py":
    project_root = os.path.abspath(os.path.join(project_root, os.pardir, os.pardir))

# Cargar el archivo de variables de entorno
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
# Leer API KEY DE DATA ALCHEMY
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

def get_transaction_by_hash(tx_hash):
    """
    Obtiene los detalles de una transacción específica usando su hash.
    
    Args:
        tx_hash (str): Hash de la transacción a buscar
        
    Returns:
        dict: Detalles de la transacción o None si no se encuentra
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getTransactionByHash",
        "params": [tx_hash]
    }
    try:
        response = requests.post(ALCHEMY_URL, json=payload)
        result = response.json().get("result")
        if result:
            # Convertir la transacción al formato esperado por extract_features
            return {
                "from": result.get("from"),
                "to": result.get("to"),
                "value": int(result.get("value", "0"), 16) / 1e18,
                "metadata": {
                    "blockTimestamp": result.get("blockTimestamp")
                },
                "category": "external"
            }
        return None
    except Exception as e:
        print(f"Error al obtener la transacción: {e}")
        return None

# === PROCESAMIENTO DE FEATURES ===

def extract_features(transfers, recent_tx=None):
    """
    Extrae features de las transacciones.
    
    Args:
        transfers: Lista de todas las transacciones (históricas + recientes)
        recent_tx: Lista de transacciones recientes a evaluar (si es None, se usan todas las transfers)
    """
    # Primero procesamos todas las transacciones para obtener el historial
    accounts = defaultdict(lambda: {
        "sent_values": [], "received_values": [],
        "sent_addresses": [], "received_addresses": [],
        "timestamps_sent": [], "timestamps_received": [],
        "erc20_sent_values": [], "erc20_received_values": [],
        "erc20_sent_tokens": [], "erc20_received_tokens": [],
        "erc20_sent_to": [], "erc20_received_from": [],
        "erc20_sent_to_contracts": [],
        "contract_sent_values": [],
        "created_contracts": 0,
        "erc20_timestamps_sent": [], "erc20_timestamps_received": [],
        "erc20_contract_timestamps": []
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
                acc["erc20_timestamps_sent"].append(timestamp)
                if to_addr and is_contract_address(to_addr):
                    acc["erc20_sent_to_contracts"].append(to_addr)
                    acc["erc20_contract_timestamps"].append(timestamp)

        if to_addr:
            acc = accounts[to_addr]
            acc["received_values"].append(value)
            acc["received_addresses"].append(from_addr)
            acc["timestamps_received"].append(timestamp)

            if category == "erc20":
                acc["erc20_received_values"].append(value)
                acc["erc20_received_tokens"].append(asset)
                acc["erc20_received_from"].append(from_addr)
                acc["erc20_timestamps_received"].append(timestamp)

    def avg_time_diff(times):
        if len(times) < 2:
            return 0
        times = sorted(pd.to_datetime(times))
        diffs = [(t2 - t1).total_seconds() / 60 for t1, t2 in zip(times[:-1], times[1:])]
        return sum(diffs) / len(diffs) if diffs else 0

    def get_most_common_token(tokens):
        if not tokens:
            return "none"
        return max(set(tokens), key=tokens.count)

    # Ahora generamos features para cada transacción reciente
    features = []
    transactions_to_process = recent_tx if recent_tx is not None else transfers
    
    for tx in transactions_to_process:
        from_addr = tx.get("from")
        if not from_addr:
            continue
            
        data = accounts[from_addr]
        sent_times = [ts for ts in data["timestamps_sent"] if ts]
        rec_times = [ts for ts in data["timestamps_received"] if ts]
        all_times = pd.to_datetime(sent_times + rec_times)
        
        erc20_sent_times = [ts for ts in data["erc20_timestamps_sent"] if ts]
        erc20_rec_times = [ts for ts in data["erc20_timestamps_received"] if ts]
        erc20_contract_times = [ts for ts in data["erc20_contract_timestamps"] if ts]

        features.append({
            "Address": from_addr,
            "Avg min between sent tnx": avg_time_diff(sent_times),
            "Avg min between received tnx": avg_time_diff(rec_times),
            "Time Diff between first and last (Mins)": (all_times.max() - all_times.min()).total_seconds() / 60 if len(all_times) > 1 else 0,
            "Sent tnx": len(data["sent_values"]),
            "Received Tnx": len(data["received_values"]),
            "Number of Created Contracts": data["created_contracts"],
            "Unique Received From Addresses": len(set(data["received_addresses"])),
            "Unique Sent To Addresses": len(set(data["sent_addresses"])),
            "min value received": min(data["received_values"], default=0),
            "max value received": max(data["received_values"], default=0),
            "avg val received": sum(data["received_values"]) / len(data["received_values"]) if data["received_values"] else 0,
            "min val sent": min(data["sent_values"], default=0),
            "max val sent": max(data["sent_values"], default=0),
            "avg val sent": sum(data["sent_values"]) / len(data["sent_values"]) if data["sent_values"] else 0,
            "min value sent to contract": min(data["contract_sent_values"], default=0),
            "max val sent to contract": max(data["contract_sent_values"], default=0),
            "avg value sent to contract": sum(data["contract_sent_values"]) / len(data["contract_sent_values"]) if data["contract_sent_values"] else 0,
            "total transactions (including tnx to create contract": len(data["sent_values"]) + len(data["received_values"]) + data["created_contracts"],
            "total Ether sent": sum(data["sent_values"]),
            "total ether received": sum(data["received_values"]),
            "total ether sent contracts": sum(data["contract_sent_values"]),
            "total ether balance": get_eth_balance(from_addr),
            "Total ERC20 tnxs": len(data["erc20_sent_values"]) + len(data["erc20_received_values"]),
            "ERC20 total Ether received": sum(data["erc20_received_values"]),
            "ERC20 total ether sent": sum(data["erc20_sent_values"]),
            "ERC20 total Ether sent contract": sum(data["erc20_sent_values"]),
            "ERC20 uniq sent addr": len(set(data["erc20_sent_to"])),
            "ERC20 uniq rec addr": len(set(data["erc20_received_from"])),
            "ERC20 uniq sent addr.1": len(set(data["erc20_sent_to"])),
            "ERC20 uniq rec contract addr": len(set(data["erc20_sent_to_contracts"])),
            "ERC20 uniq sent contract addr": len(set(data["erc20_sent_to_contracts"])),
            "ERC20 uniq sent token name": len(set(data["erc20_sent_tokens"])),
            "ERC20 uniq rec token name": len(set(data["erc20_received_tokens"])),
            "ERC20 most sent token type": get_most_common_token(data["erc20_sent_tokens"]),
            "ERC20_most_rec_token_type": get_most_common_token(data["erc20_received_tokens"]),
            "ERC20 avg time between sent tnx": avg_time_diff(erc20_sent_times),
            "ERC20 avg time between rec tnx": avg_time_diff(erc20_rec_times),
            "ERC20 avg time between rec 2 tnx": avg_time_diff(erc20_rec_times),
            "ERC20 avg time between contract tnx": avg_time_diff(erc20_contract_times),
            "ERC20 min val rec": min(data["erc20_received_values"], default=0),
            "ERC20 max val rec": max(data["erc20_received_values"], default=0),
            "ERC20 avg val rec": sum(data["erc20_received_values"]) / len(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20 min val sent": min(data["erc20_sent_values"], default=0),
            "ERC20 max val sent": max(data["erc20_sent_values"], default=0),
            "ERC20 avg val sent": sum(data["erc20_sent_values"]) / len(data["erc20_sent_values"]) if data["erc20_sent_values"] else 0,
            "ERC20 min val sent contract": min(data["erc20_sent_values"], default=0),
            "ERC20 max val sent contract": max(data["erc20_sent_values"], default=0),
            "ERC20 avg val sent contract": sum(data["erc20_sent_values"]) / len(data["erc20_sent_values"]) if data["erc20_sent_values"] else 0,
            "ERC20 min val rec contract": min(data["erc20_received_values"], default=0),
            "ERC20 max val rec contract": max(data["erc20_received_values"], default=0),
            "ERC20 avg val rec contract": sum(data["erc20_received_values"]) / len(data["erc20_received_values"]) if data["erc20_received_values"] else 0,
            "ERC20 sent tnx": len(data["erc20_sent_values"]),
            "ERC20 contracts sent tnx": len([v for v in data["erc20_sent_values"] if v > 0]),
            "ERC20 rec tnx": len(data["erc20_received_values"]),
            "ERC20 contracts rec tnx": len([v for v in data["erc20_received_values"] if v > 0]),
            "ERC20 most sent token rec": get_most_common_token(data["erc20_sent_tokens"]),
            "ERC20 most rec token rec": get_most_common_token(data["erc20_received_tokens"])
        })

    return pd.DataFrame(features)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de transacciones de Ethereum")
    parser.add_argument("--minutes", type=int, help="Cantidad de minutos hacia atrás para obtener transacciones recientes")
    parser.add_argument("--max_tx", type=int, default=10, help="Máximo número de transacciones por dirección")
    parser.add_argument("--tx_hash", type=str, help="Hash de la transacción específica a analizar")
    args = parser.parse_args()

    total_start_time = time.time()

    if args.tx_hash:
        print(f"▶ Buscando transacción específica con hash: {args.tx_hash}")
        start_time = time.time()
        tx = get_transaction_by_hash(args.tx_hash)
        if not tx:
            print("❌ No se encontró la transacción especificada")
            exit(1)
        
        print(f"⏱️ Tiempo de búsqueda de transacción: {round(time.time() - start_time, 2)} segundos")
        
        addresses = set()
        if tx.get("from"):
            addresses.add(tx["from"])
            print(f"📤 Remitente: {tx['from']}")
        else:
            print("❌ No se encontró la dirección del remitente")
            exit(1)
        
        print(f"🧾 Analizando 1 dirección.")
        
        recent_tx = [tx]
        all_hist_txs = [tx]
    else:
        if not args.minutes:
            print("❌ Se requiere especificar --minutes o --tx_hash")
            exit(1)
            
        print(f"▶ Descargando transacciones recientes de los últimos {args.minutes} minutos...")
        start_time = time.time()
        recent_tx = get_recent_transfers(minutes=args.minutes, max_tx=args.max_tx)
        print(f"⏱️ Tiempo de descarga de transacciones recientes: {round(time.time() - start_time, 2)} segundos")
        print(f"📥 Se obtuvieron {len(recent_tx)} transacciones.")

        print("\n📋 Hashes de las transacciones encontradas:")
        for tx in recent_tx:
            print(f"  • {tx.get('hash', 'Hash no disponible')}")

        addresses = set()
        for tx in recent_tx:
            if tx.get("from"):
                addresses.add(tx["from"])
        print(f"🧾 {len(addresses)} direcciones de remitentes encontradas.")
        
        all_hist_txs = recent_tx.copy()

    print("📚 Consultando históricos por dirección...")
    start_time = time.time()
    for i, addr in enumerate(addresses):
        print(f"[{i+1}/{len(addresses)}] Consultando histórico para {addr}")
        hist_txs = get_historical_transfers_for_address(addr, max_tx=args.max_tx)
        all_hist_txs.extend(hist_txs)
        time.sleep(0.25)  # Evitar rate limiting
    print(f"⏱️ Tiempo total de consulta histórica: {round(time.time() - start_time, 2)} segundos")
    print(f"📦 Total de transacciones históricas recopiladas: {len(all_hist_txs)}")

    print("🔍 Extrayendo features...")
    start_time = time.time()
    df_features = extract_features(all_hist_txs, recent_tx)
    print(f"⏱️ Tiempo de extracción de features: {round(time.time() - start_time, 2)} segundos")
    print(f"📊 Features extraídos para {len(df_features)} transacciones recientes")
    
    # Generar nombre de archivo descriptivo
    if args.tx_hash:
        output_file = f"features_tx_{args.tx_hash[:10]}.csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"features_recent_{args.minutes}m_{args.max_tx}tx_{timestamp}.csv"

    # Construir la ruta completa al archivo de salida
    if "get_data" in project_root:
        project_root = os.path.join(project_root, os.pardir, os.pardir)
    output_file = os.path.join(project_root, "features_downloads", output_file)
    
    # Asegurarse de que el directorio existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_features.to_csv(output_file, index=False)
    print(f"✅ Features guardadas en '{output_file}'")

    total_elapsed = round(time.time() - total_start_time, 2)
    print(f"🏁 Proceso completo finalizado en {total_elapsed} segundos.")
