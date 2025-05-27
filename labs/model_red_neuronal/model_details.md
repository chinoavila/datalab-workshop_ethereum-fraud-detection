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

## Rendimiento (Actualizado Mayo 2025)

### Métricas en Datos de Prueba
- Accuracy: 85.2%
- Precision: 81.7%
- Recall: 79.3%
- F1-Score: 80.5%
- AUC-ROC: 0.895
- Tiempo de inferencia promedio: 5.6ms por muestra

### Ventajas
- Implementación personalizada que permite ajustes de bajo nivel
- Mayor control sobre el proceso de entrenamiento
- Menor dependencia de frameworks externos
- Apropiado para entender los fundamentos de redes neuronales

### Limitaciones
- Rendimiento inferior a modelos basados en árboles para este dataset
- Mayor complejidad de mantenimiento
- Tiempo de entrenamiento más largo
- Requiere ajuste manual de muchos hiperparámetros

### Casos de Uso Específicos
- Análisis académico del funcionamiento de redes neuronales
- Escenarios donde se requiere control total sobre la arquitectura
- Entornos con limitaciones de dependencias externas

## Resultados de Análisis de Sensibilidad
El modelo es particularmente sensible a:
1. Tasa de aprendizaje
2. Número de épocas
3. Tamaño de las capas ocultas
4. Dropout rate

## Última Actualización
- Fecha: Mayo 2025
- Estado: Mantenimiento
- Uso recomendado: Académico/Experimental
