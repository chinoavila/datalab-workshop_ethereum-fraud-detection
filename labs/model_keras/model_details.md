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

## Rendimiento (Actualizado Mayo 2025)

### Métricas en Datos de Prueba
- Accuracy: 86.7%
- Precision: 82.4% 
- Recall: 80.9%
- F1-Score: 81.6%
- AUC-ROC: 0.908
- Tiempo de inferencia promedio: 8.3ms por muestra

### Análisis de Rendimiento
El modelo Keras muestra un rendimiento superior al de la red neuronal personalizada, pero sigue siendo inferior a los modelos basados en árboles (Random Forest y XGBoost) para este conjunto de datos específico. Sin embargo, el uso de early stopping y batch normalization ha demostrado ser efectivo para evitar el sobreajuste.

### Ventajas
- Framework maduro y bien mantenido
- Fácil experimentación con diferentes arquitecturas
- Buen equilibrio entre flexibilidad y facilidad de uso
- Integración con TensorFlow para optimización avanzada
- Potencial de mejora mediante ajuste de hiperparámetros

### Limitaciones
- Mayor tiempo de inferencia comparado con modelos basados en árboles
- Requiere más recursos computacionales
- Menor interpretabilidad de las decisiones del modelo
- Sensibilidad al inicializado de pesos aleatorios

## Mejoras Recientes
- Implementación de capas de regularización adicionales
- Optimización del learning rate con scheduler dinámico
- Experimentación con arquitecturas más profundas

## Última Actualización
- Fecha: Mayo 2025
- Estado: Productivo - Secundario
- Mantenimiento: Activo
