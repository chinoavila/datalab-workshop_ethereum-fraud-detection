# Obtención de Datos de Ethereum usando Alchemy

Este módulo está diseñado para interactuar con la API de Alchemy y obtener datos relevantes de transacciones de Ethereum. Los datos obtenidos incluyen detalles sobre transacciones recientes, balances de direcciones, y features relacionados con el comportamiento de las adressess, como el número de transacciones enviadas y recibidas, el valor total enviado/recibido, entre otros.

El objetivo de este proceso es extraer features que luego serán utilizadas en modelos para hacer predicciones sobre posibles fraudes.

## Requisitos

Este módulo depende de las siguientes librerías:

- `requests`: Para realizar solicitudes HTTP a la API de Alchemy.
- `pandas`: Para manipular los datos de las transacciones y generar el archivo de características.
- `python-dotenv`: Para cargar variables de entorno desde el archivo `.env`.

Puedes instalar todas las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Configuración

Es necesario configurar la clave API de Alchemy para poder hacer las solicitudes. Esto se realiza mediante el archivo `.env`, que debe estar en el directorio raíz del proyecto.

Ejemplo de archivo `.env`:

```bash
ALCHEMY_API_KEY=tu_clave_api_de_alchemy_aqui
```

## Funcionalidad

El script proporciona varias funciones para interactuar con la API de Alchemy y obtener los datos relevantes:

1. Obtención de balances de direcciones
   La función `get_eth_balance(address)` obtiene el saldo en Ether de una dirección específica.

2. Detección de direcciones de contratos inteligentes
   La función `is_contract_address(address)` permite verificar si una dirección es un contrato inteligente. Esto es útil para filtrar las transacciones que involucran contratos.

3. Obtención de transacciones recientes
   La función `get_recent_transfers(minutes=1, max_tx=10)` obtiene las transacciones recientes en la blockchain de Ethereum en el intervalo de tiempo especificado (en minutos). Se limita el número de transacciones recuperadas a `max_tx`.

4. Obtención de transacciones históricas por dirección
   La función `get_historical_transfers_for_address(address, max_tx=100)` obtiene las transacciones históricas de una dirección específica. También tiene un límite máximo de transacciones a recuperar.

5. Extracción de features
   La función `extract_features(transfers)` toma las transacciones obtenidas y las procesa para extraer un conjunto de características relevantes. Estas características incluyen:

- Promedio de minutos entre transacciones enviadas y recibidas.
- Número de transacciones enviadas y recibidas.
- Valor total de Ether enviado y recibido.
- Número de contratos creados por la dirección.
- Datos sobre transacciones ERC20, como valores enviados/recibidos y tokens utilizados.

## Ejecución del Script

El script realiza lo siguiente:

- Descarga las transacciones recientes de la blockchain de Ethereum.
- Obtiene el historial de transacciones de todas las direcciones involucradas en esas transacciones.
- Extrae las características relevantes de las transacciones históricas.
- Guarda estas características en un archivo CSV llamado historical_features_eth.csv.

Ejemplo de ejecución:

```bash
python3 tu_script.py
```

Este script descargará las transacciones y generará un archivo CSV con las características de las direcciones involucradas en dichas transacciones.
