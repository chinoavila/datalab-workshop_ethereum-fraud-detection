# Random Forest para Detección de Fraude

## Descripción
Implementación de un modelo Random Forest para la detección de transacciones fraudulentas en Ethereum, optimizado para manejar datos desbalanceados.

## Características del Modelo
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: balanced

## Preprocesamiento
- Normalización de variables numéricas
- Manejo de valores faltantes
- Codificación de variables categóricas
- Balanceo de clases usando SMOTE

## Selección de Características
- Importancia de características usando feature_importances_
- Selección basada en correlación
- Eliminación recursiva de características (RFE)

## Métricas de Evaluación
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Validación
- Cross-validation con 5 folds
- Validación temporal (train/test split por fecha)

## Dependencias
Ver `requirements.txt` para la lista completa de dependencias.

## Uso
```python
python RF_ether.py
```

## Outputs
- Modelo guardado en formato .joblib
- Gráficas de importancia de características
- Matriz de confusión
- Reporte de métricas
