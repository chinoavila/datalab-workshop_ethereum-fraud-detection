# Detalles del Modelo de Regresión Logística

Este documento proporciona una descripción detallada de la implementación del modelo de regresión logística para la detección de fraude en transacciones de Ethereum.

## Funcionalidades Principales

### 1. Carga y Preparación de Datos
- Lectura del conjunto de datos desde archivo CSV ('transaction_dataset.csv')
- Preprocesamiento de datos:
  - Codificación one-hot de variables categóricas
  - Manejo de valores nulos mediante eliminación
  - División en conjuntos de entrenamiento (30%) y prueba (70%)

### 2. Implementación del Modelo
- Utilización de LogisticRegression de scikit-learn con los siguientes parámetros:
  - C=1.0 (inverso de la regularización)
  - penalty='l2' (regularización L2)
  - solver="newton-cg" (optimizador)
  - Opción de usar class_weight="balanced" para datos desbalanceados

### 3. Estrategias de Manejo de Datos Desbalanceados
Se implementan varias técnicas para manejar el desbalance de clases:
- Under-sampling con NearMiss
- Over-sampling con RandomOverSampler
- Combinación SMOTE-Tomek
- Balanced Bagging Classifier para ensamble de modelos
- Penalización por clase mediante class_weight

### 4. Evaluación del Modelo
El desempeño se evalúa usando:
- Matriz de confusión visualizada con seaborn
- Métricas detalladas:
  - Precisión (precision)
  - Sensibilidad (recall)
  - F1-score
  - Accuracy general y por clase

## Dependencias
- Python 3.x
- Bibliotecas principales:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - imblearn

## Instrucciones de Uso
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Asegurarse de tener el dataset en la ruta correcta
3. Ejecutar el notebook o script para entrenar y evaluar el modelo

## Resultados y Métricas
El modelo logra los siguientes resultados (con la mejor configuración usando BalancedBaggingClassifier):
- Accuracy: 1.00
- Precision Clase 0 (No Fraude): 1.00
- Precision Clase 1 (Fraude): 1.00
- F1-score promedio: 1.00