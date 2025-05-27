# Extracción de Features de Transacciones Ethereum

Este script permite analizar transacciones de Ethereum, ya sea transacciones recientes o una transacción específica, y extraer features relevantes para la detección de fraude.

## Características Principales

- Análisis de transacciones recientes (últimos X minutos)
- Análisis de una transacción específica por su hash
- Extracción de features históricos de las direcciones involucradas
- Generación de archivos CSV con features detallados

## Requisitos

- Python 3.6+
- API Key de Alchemy (configurada en el archivo .env)
- Dependencias: requests, pandas, python-dotenv

## Uso

### Análisis de Transacciones Recientes

```bash
python generate_eth_features_history.py --minutes <minutos> [--max_tx <número>]
```

Ejemplo:

```bash
python generate_eth_features_history.py --minutes 5 --max_tx 20
```

### Análisis de una Transacción Específica

```bash
python generate_eth_features_history.py --tx_hash <hash_de_la_transacción>
```

Ejemplo:

```bash
python generate_eth_features_history.py --tx_hash 0x123...
```

## Parámetros

- `--minutes`: (Obligatorio para análisis reciente) Número de minutos hacia atrás para obtener transacciones
- `--max_tx`: (Opcional) Máximo número de transacciones históricas a analizar por dirección (default: 10)
- `--tx_hash`: (Obligatorio para análisis específico) Hash de la transacción a analizar

## Formato de Salida

El script genera un archivo CSV con las siguientes columnas:

### Features Básicos

- `Address`: Dirección del remitente
- `Avg min between sent tnx`: Promedio de minutos entre transacciones enviadas
- `Avg min between received tnx`: Promedio de minutos entre transacciones recibidas
- `Time Diff between first and last (Mins)`: Diferencia de tiempo entre primera y última transacción
- `Sent tnx`: Número de transacciones enviadas
- `Received Tnx`: Número de transacciones recibidas
- `Number of Created Contracts`: Número de contratos creados
- `Unique Received From Addresses`: Número de direcciones únicas de las que se recibió
- `Unique Sent To Addresses`: Número de direcciones únicas a las que se envió

### Features de Valores

- `min value received`: Valor mínimo recibido
- `max value received`: Valor máximo recibido
- `avg val received`: Valor promedio recibido
- `min val sent`: Valor mínimo enviado
- `max val sent`: Valor máximo enviado
- `avg val sent`: Valor promedio enviado
- `min value sent to contract`: Valor mínimo enviado a contratos
- `max val sent to contract`: Valor máximo enviado a contratos
- `avg value sent to contract`: Valor promedio enviado a contratos

### Totales

- `total transactions`: Total de transacciones (incluyendo creación de contratos)
- `total Ether sent`: Total de Ether enviado
- `total ether received`: Total de Ether recibido
- `total ether sent contracts`: Total de Ether enviado a contratos
- `total ether balance`: Balance total de Ether

### Features ERC20

- `Total ERC20 tnxs`: Total de transacciones ERC20
- `ERC20 total Ether received`: Total de Ether recibido en transacciones ERC20
- `ERC20 total ether sent`: Total de Ether enviado en transacciones ERC20
- `ERC20 total Ether sent contract`: Total de Ether enviado a contratos en transacciones ERC20
- `ERC20 uniq sent addr`: Número de direcciones únicas a las que se envió ERC20
- `ERC20 uniq rec addr`: Número de direcciones únicas de las que se recibió ERC20
- `ERC20 uniq sent addr.1`: Número de direcciones únicas a las que se envió ERC20 (duplicado para compatibilidad)
- `ERC20 uniq rec contract addr`: Número de contratos únicos a los que se envió ERC20

### Tiempos ERC20

- `ERC20 avg time between sent tnx`: Promedio de tiempo entre transacciones ERC20 enviadas
- `ERC20 avg time between rec tnx`: Promedio de tiempo entre transacciones ERC20 recibidas
- `ERC20 avg time between rec 2 tnx`: Promedio de tiempo entre transacciones ERC20 recibidas (duplicado para compatibilidad)
- `ERC20 avg time between contract tnx`: Promedio de tiempo entre transacciones ERC20 a contratos

### Valores ERC20

- `ERC20 min val rec`: Valor mínimo recibido en ERC20
- `ERC20 max val rec`: Valor máximo recibido en ERC20
- `ERC20 avg val rec`: Valor promedio recibido en ERC20
- `ERC20 min val sent`: Valor mínimo enviado en ERC20
- `ERC20 max val sent`: Valor máximo enviado en ERC20
- `ERC20 avg val sent`: Valor promedio enviado en ERC20
- `ERC20 min val sent contract`: Valor mínimo enviado a contratos en ERC20
- `ERC20 max val sent contract`: Valor máximo enviado a contratos en ERC20
- `ERC20 avg val sent contract`: Valor promedio enviado a contratos en ERC20

### Tokens ERC20

- `ERC20 uniq sent token name`: Número de tokens ERC20 únicos enviados
- `ERC20 uniq rec token name`: Número de tokens ERC20 únicos recibidos
- `ERC20 most sent token type`: Token ERC20 más común enviado
- `ERC20_most_rec_token_type`: Token ERC20 más común recibido

## Nombres de Archivos de Salida

- Para transacciones recientes: `features_recent_<minutos>m_<max_tx>tx_<timestamp>.csv`
- Para transacción específica: `features_tx_<hash>.csv`

## Notas Importantes

1. El script analiza las transacciones recientes y genera una fila por cada transacción, incluyendo los features históricos del remitente.
2. Los features históricos se calculan usando todas las transacciones disponibles hasta el momento de la transacción actual.
3. El script incluye un delay de 0.25 segundos entre consultas para evitar rate limiting.
4. Se requiere una API key de Alchemy configurada en el archivo .env.

## Ejemplo de Uso

```bash
# Analizar transacciones de los últimos 5 minutos
python generate_eth_features_history.py --minutes 5 --max_tx 20

# Analizar una transacción específica
python generate_eth_features_history.py --tx_hash 0x123...
```

## Salida de Consola

El script muestra información detallada sobre el proceso:

```
▶ Descargando transacciones recientes de los últimos 5 minutos...
⏱️ Tiempo de descarga de transacciones recientes: 1.23 segundos
📥 Se obtuvieron 5 transacciones.

📋 Hashes de las transacciones encontradas:
  • 0x123...
  • 0x456...
  ...

🧾 3 direcciones de remitentes encontradas.
📚 Consultando históricos por dirección...
[1/3] Consultando histórico para 0x789...
[2/3] Consultando histórico para 0xabc...
[3/3] Consultando histórico para 0xdef...
⏱️ Tiempo total de consulta histórica: 2.45 segundos
📦 Total de transacciones históricas recopiladas: 45

🔍 Extrayendo features...
⏱️ Tiempo de extracción de features: 0.5 segundos
📊 Features extraídos para 5 transacciones recientes

✅ Features guardadas en 'features_recent_5m_20tx_20240315_123456.csv'
🏁 Proceso completo finalizado en 4.18 segundos.
```
