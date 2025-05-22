# Modelo Keras para Detección de Fraude

## Descripción
Implementación de una red neuronal profunda usando Keras/TensorFlow para la detección de transacciones fraudulentas en Ethereum.

## Arquitectura
- Capa de entrada: Dense(128, activation='relu')
- Capas ocultas:
  - Dense(64, activation='relu')
  - Dropout(0.3)
  - Dense(32, activation='relu')
  - Dropout(0.2)
- Capa de salida: Dense(1, activation='sigmoid')

## Características
- Normalización de datos usando StandardScaler
- Early stopping para prevenir overfitting
- Batch normalization entre capas
- Optimizador Adam con learning rate adaptativo

## Métricas de Rendimiento
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Hiperparámetros
- Batch size: 32
- Epochs: 100 (con early stopping)
- Learning rate inicial: 0.001
- Validation split: 0.2

## Dependencias
Ver `requirements.txt` para la lista completa de dependencias.

## Uso
```python
python Keras_ether.py
```

## Outputs
- Modelo guardado en formato .joblib
- Gráficas de entrenamiento
- Reporte de métricas
