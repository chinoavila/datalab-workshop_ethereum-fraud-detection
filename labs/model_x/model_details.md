# Detalles del Modelo de Regresión Logística

Este documento proporciona una descripción detallada del archivo `model.py`, que implementa el modelo de regresión logística para la detección de fraude en transacciones de Ethereum.

## Funcionalidades Principales

### 1. Carga de Datos
El script incluye funciones para:
- Leer el conjunto de datos desde archivos CSV.
- Preprocesar los datos, incluyendo la limpieza, normalización y manejo de valores faltantes.

### 2. Entrenamiento del Modelo
- Implementación del algoritmo de regresión logística utilizando bibliotecas como `scikit-learn`.
- División del conjunto de datos en subconjuntos de entrenamiento y prueba.
- Ajuste de hiperparámetros para optimizar el rendimiento del modelo.

### 3. Evaluación del Modelo
El desempeño del modelo se mide utilizando las siguientes métricas:
- **Precisión**: Proporción de predicciones correctas sobre el total de predicciones.
- **Recall**: Capacidad del modelo para identificar correctamente las instancias positivas.
- **F1-Score**: Media armónica de precisión y recall, útil para conjuntos de datos desbalanceados.

### 4. Visualización
- Generación de gráficos como curvas ROC y matrices de confusión para evaluar el rendimiento del modelo.
- Tablas que resumen las métricas clave.

## Modularidad
El archivo está diseñado para ser modular, lo que permite:
- Extender fácilmente las funcionalidades existentes.
- Integrar nuevas características o algoritmos.

## Requisitos
- **Python 3.8 o superior**
- Bibliotecas necesarias (ver archivo `requirements.txt` en la misma carpeta).

## Instrucciones de Uso
1. Asegúrate de que las dependencias estén instaladas:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el script `model.py` para entrenar y evaluar el modelo.

## Notas Adicionales
Este modelo es una implementación básica y puede mejorarse añadiendo técnicas avanzadas como:
- Selección de características.
- Métodos de regularización como L1 y L2.
- Validación cruzada para evaluar la robustez del modelo.