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

## Rendimiento (Actualizado Mayo 2025)

### Métricas en Datos de Prueba
- Accuracy: 90.6%
- Precision: 87.9%
- Recall: 86.4%
- F1-Score: 87.1%
- AUC-ROC: 0.948
- Tiempo de inferencia promedio: 2.8ms por muestra

### Características Importantes
El modelo identificó las siguientes características como las más predictivas:
1. Unique_Sent_To_Addresses
2. Time_Diff_between_first_and_last(Mins)
3. Ratio_Sent_Received
4. ERC20_Total_Tnx
5. Avg_min_between_sent_tnx

### Ventajas
- Alto rendimiento general
- Excelente equilibrio entre precision y recall
- Velocidad de inferencia optimizada
- Buena capacidad de generalización
- Manejo intrínseco de datos dispersos y missing values

### Limitaciones
- Mayor complejidad y menor interpretabilidad que Random Forest
- Requiere más ajuste fino de hiperparámetros

## Casos de Uso Recomendados
- Análisis por lotes de grandes volúmenes de transacciones
- Sistemas de puntuación de riesgo en tiempo real
- Complemento a sistemas basados en reglas

## Última Actualización
- Fecha: Mayo 2025
- Estado: Productivo
- Mantenimiento: Activo
