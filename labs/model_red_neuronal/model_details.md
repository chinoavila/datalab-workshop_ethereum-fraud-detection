# Red Neuronal Personalizada para Detección de Fraude

## Descripción
Implementación de una red neuronal personalizada para la detección de transacciones fraudulentas en Ethereum, con arquitectura y funciones de activación optimizadas.

## Arquitectura
- Capa de entrada: 64 neuronas, ReLU
- Primera capa oculta: 32 neuronas, ReLU
- Segunda capa oculta: 16 neuronas, ReLU
- Capa de salida: 1 neurona, Sigmoid

## Características
- Implementación desde cero sin frameworks de alto nivel
- Backpropagation personalizado
- Optimizador SGD con momentum
- Regularización L2
- Dropout personalizado

## Preprocesamiento
- Normalización usando Z-score
- Codificación one-hot para variables categóricas
- Balanceo de datos usando under-sampling

## Hiperparámetros
- Learning rate: 0.001
- Momentum: 0.9
- Batch size: 32
- Epochs: 50
- Dropout rate: 0.3

## Dependencias
Ver `requirements.txt` para la lista completa de dependencias.

## Uso
```python
python Redes_Neuronales_ether.py
```

## Outputs
- Modelo guardado en formato .joblib
- Gráficas de convergencia
- Análisis de sensibilidad
- Reporte de rendimiento
