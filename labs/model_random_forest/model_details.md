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

## Rendimiento (Actualizado Mayo 2025)

### Métricas en Datos de Prueba
- Accuracy: 91.3%
- Precision: 88.5%
- Recall: 87.2%
- F1-Score: 87.8%
- AUC-ROC: 0.952
- Tiempo de inferencia promedio: 3.2ms por muestra

### Características Importantes
Según el análisis de importancia de características, las más relevantes son:
1. Time_Diff_between_first_and_last(Mins)
2. Unique_Received_From_Addresses
3. Unique_Sent_To_Addresses
4. Ratio_Received_Sent 
5. Total_Transactions (Sent + Received)

### Ventajas
- Mejor rendimiento general entre todos los modelos
- Alta capacidad para detectar patrones complejos
- Robustez ante valores atípicos
- Menor tendencia al sobreajuste que XGBoost
- Buena interpretabilidad a través del análisis de importancia de características

### Limitaciones
- Ligeramente más lento que XGBoost en inferencia
- Mayor uso de memoria para modelos grandes
- Menos eficiente con datos muy dispersos

## Estado Actual
Este es el modelo recomendado para producción debido a su combinación óptima de precisión, rendimiento y robustez.

## Última Actualización
- Fecha: Mayo 2025
- Estado: Productivo - Modelo Principal
- Mantenimiento: Activo
