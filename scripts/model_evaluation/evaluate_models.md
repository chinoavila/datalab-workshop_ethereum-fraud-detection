# Evaluación de Modelos de Detección de Fraude

Este script realiza la evaluación y comparación de diferentes modelos de machine learning entrenados para la detección de fraude en transacciones de Ethereum.

## Descripción

El script `evaluate_models.py` está diseñado para cargar y evaluar múltiples modelos de machine learning, incluyendo:
- Random Forest
- Keras (Red Neuronal)
- Regresión Logística
- Red Neuronal Personalizada
- XGBoost

## Funcionalidades Principales

### 1. Carga de Datos y Modelos
- Busca automáticamente el dataset más reciente de características
- Carga los modelos y scalers guardados
- Verifica y asegura la consistencia de las columnas

### 2. Predicción y Evaluación
- Realiza predicciones con cada modelo
- Calcula probabilidades de fraude
- Mide tiempos de ejecución
- Genera diagnósticos detallados

### 3. Visualización y Reportes
- Crea histogramas de distribución de probabilidades
- Genera reportes detallados de predicciones
- Produce resúmenes comparativos
- Guarda resultados en formato CSV

## Estructura de Resultados

El script genera los siguientes archivos en el directorio `resultados/[nombre_dataset]/`:
- `predicciones.csv`: Predicciones detalladas de todos los modelos
- `resumen_predicciones.csv`: Comparativa de rendimiento entre modelos
- `distribucion_probabilidades.png`: Visualización de las distribuciones
- `diagnostico.txt`: Información detallada de diagnóstico

## Métricas Evaluadas
- Número de fraudes detectados
- Porcentaje de fraudes en el dataset
- Tiempos de ejecución
- Distribución de probabilidades
- Estadísticas descriptivas de las predicciones

## Requisitos

Ver `requirements.txt` para la lista completa de dependencias.

## Uso

```python
python evaluate_models.py
```

El script automáticamente:
1. Detecta el dataset más reciente
2. Carga todos los modelos disponibles
3. Realiza predicciones
4. Genera visualizaciones y reportes
5. Guarda los resultados en el directorio correspondiente

## Notas Importantes

- Asegúrese de que todos los modelos y scalers estén presentes en el directorio `models/`
- El script espera un formato específico en las columnas del dataset
- Las predicciones se normalizan al rango [0,1] para comparabilidad
- Se recomienda revisar el archivo de diagnóstico para detectar posibles anomalías
