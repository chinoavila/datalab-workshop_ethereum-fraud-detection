# Datalab Workshop - Proyecto de Detección de Fraude con Dataset de Transacciones con Ethereum

## Descripción
Este proyecto tiene como objetivo detectar actividades fraudulentas en un conjunto de datos de transacciones realizadas con Ethereum. Utilizamos técnicas avanzadas de ciencia de datos y aprendizaje automático para analizar patrones y comportamientos sospechosos.

## Estructura del Proyecto

```
datalab-workshop_ethereum-fraud-detection/
├── common_functions/            # Funciones compartidas entre modelos
│   ├── data_utils.py           # Utilidades para procesamiento de datos
│   ├── eval_utils.py           # Funciones de evaluación
│   ├── balance_utils.py        # Utilidades para manejo de datos desbalanceados
│   └── requirements.txt        # Dependencias comunes
├── datasets/                   # Datasets del proyecto
│   ├── transaction_dataset.csv     # Dataset original
│   └── transaction_dataset_clean.csv # Dataset procesado
├── models/                     # Modelos entrenados
│   ├── keras_model.joblib      # Modelo Keras guardado
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── red_neuronal_model.joblib
│   ├── xgboost_model.joblib
│   └── *_scaler.joblib        # Escaladores para cada modelo
├── labs/                      # Implementación de modelos
│   ├── EDA/                   # Análisis Exploratorio de Datos
│   ├── model_keras/           # Modelo con Keras
│   ├── model_logistic_regression/
│   ├── model_random_forest/
│   ├── model_red_neuronal/
│   └── model_xgboost/
├── notebooks/                 # Jupyter notebooks de desarrollo
│   ├── EDA/
│   ├── model_keras/
│   ├── model_logistic_regression/
│   ├── model_random_forest/
│   └── model_red_neuronal/
├── scripts/                   # Scripts de utilidad
│   ├── get_data/             # Scripts para obtención de datos
│   └── model_evaluation/      # Scripts de evaluación de modelos
├── webapp/                    # Aplicación web Streamlit
│   ├── app.py                # Aplicación principal
│   └── requirements.txt      # Dependencias de la webapp
└── README.md
```

## Componentes Principales

### Common Functions
Módulos compartidos que implementan funcionalidades comunes:
- `data_utils.py`: Preprocesamiento y manipulación de datos
- `eval_utils.py`: Métricas y funciones de evaluación
- `balance_utils.py`: Manejo de datasets desbalanceados

### Modelos Implementados
Se han implementado varios modelos de machine learning:
- Regresión Logística
- Random Forest
- XGBoost
- Red Neuronal (Tensorflow/Keras)
- Red Neuronal Personalizada

Cada modelo está disponible en:
- Notebooks (`notebooks/model_*/`): Desarrollo y análisis
- Scripts (`labs/model_*/`): Implementación producción
- Modelos guardados (`models/`): Archivos .joblib

### WebApp
Aplicación web implementada con Streamlit que permite:
- Evaluar modelos entrenados
- Descargar y analizar datos en tiempo real
- Visualizar métricas y distribuciones
- Generar reportes de evaluación

## Requisitos

### Python y Dependencias
- Python 3.8 o superior
- Dependencias por componente:
  ```
  # Dependencias comunes (common_functions/requirements.txt)
  pandas>=1.2.4
  numpy>=1.19.2
  scikit-learn>=0.24.2

  # Dependencias WebApp (webapp/requirements.txt)
  streamlit>=1.22.0
  pandas>=1.2.4
  matplotlib>=3.4.2
  seaborn>=0.11.1

  # Dependencias notebooks (notebooks/*/requirements.txt)
  jupyter>=1.0.0
  ipykernel>=6.0.0
  plotly>=5.1.0

  # Dependencias específicas de modelos
  tensorflow>=2.6.0  # Para Keras y redes neuronales
  xgboost>=1.4.2    # Para modelo XGBoost
  ```

### Instalación
1. Clonar el repositorio:
   ```powershell
   git clone https://github.com/chinoavila/datalab-workshop_ethereum-fraud-detection
   ```

2. Crear y activar entorno virtual:
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```

3. Instalar dependencias:
   ```powershell
   pip install -r common_functions/requirements.txt
   pip install -r webapp/requirements.txt
   ```

## Uso

### Ejecutar la WebApp
```powershell
cd webapp
streamlit run app.py
```
La webapp permite:
- Evaluación en tiempo real de transacciones
- Visualización de métricas y resultados
- Descarga automática de nuevos datos
- Comparación de modelos entrenados

### Entrenar Modelos
Cada modelo puede entrenarse individualmente:
```powershell
cd labs/model_[nombre_modelo]
python model.py
```

Modelos disponibles:
- `model_keras/Keras_ether.py`
- `model_logistic_regression/model.py`
- `model_random_forest/RF_ether.py`
- `model_red_neuronal/Redes_Neuronales_ether.py`
- `model_xgboost/model.py`

### Análisis Exploratorio
```powershell
cd notebooks/EDA
jupyter notebook Transacciones_EDA.ipynb
```

### Generar Datos
Para obtener nuevos datos de transacciones:
```powershell
cd scripts/get_data
python generate_eth_features_history.py --minutes 5 --max-tx 100
```

Los datos se guardarán automáticamente en:
- Datos originales: `datasets/transaction_dataset.csv`
- Datos procesados: `datasets/transaction_dataset_clean.csv`
- Features generadas: `scripts/get_data/features_recent_*.csv`

## Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Envía un pull request

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.