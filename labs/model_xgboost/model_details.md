# XGBoost para Detección de Fraude

## Descripción
Implementación de un modelo XGBoost para la detección de transacciones fraudulentas en Ethereum, optimizado para alto rendimiento y manejo de datos desbalanceados.

## Parámetros del Modelo
- max_depth: 6
- learning_rate: 0.1
- n_estimators: 100
- objective: 'binary:logistic'
- scale_pos_weight: balanceado automáticamente
- subsample: 0.8
- colsample_bytree: 0.8

## Características
- Manejo eficiente de datos dispersos
- Poda temprana de árboles
- Regularización L1 y L2
- Procesamiento paralelo

## Preprocesamiento
- Encoding de variables categóricas
- Manejo de valores faltantes nativo
- Feature scaling opcional
- Feature selection usando SHAP values

## Cross-Validación
- K-fold estratificado
- Early stopping usando validación
- Monitoreo de métricas múltiples

## Optimización
- Grid search para hiperparámetros
- Optimización bayesiana opcional
- Learning rate scheduling

## Dependencias
Ver `requirements.txt` para la lista completa de dependencias.

## Uso
```python
python model.py
```

## Outputs
- Modelo guardado en formato .joblib
- SHAP value plots
- Feature importance plots
- Reporte detallado de métricas
