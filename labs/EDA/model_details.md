# Análisis Exploratorio de Datos (EDA)

## Descripción
Este módulo realiza un análisis exploratorio completo del conjunto de datos de transacciones de Ethereum para identificar patrones y características relevantes para la detección de fraude.

## Características Principales
- Análisis de distribuciones de variables clave
- Identificación de correlaciones entre variables
- Visualización de patrones temporales
- Análisis de comportamiento por dirección
- Detección de valores atípicos

## Visualizaciones
El script genera las siguientes visualizaciones:
1. Distribución de valores de transacciones
2. Patrones temporales de actividad
3. Relaciones entre variables numéricas
4. Análisis de direcciones más activas
5. Mapas de calor de correlaciones

## Dependencias
Ver `requirements.txt` para la lista completa de dependencias.

## Uso
```python
python Transacciones_EDA.py
```

## Outputs
- Gráficos guardados en formato PNG/PDF
- Estadísticas descriptivas en formato CSV
- Insights documentados en el notebook asociado

## Hallazgos Clave (Actualizado Mayo 2025)

### Distribución de Datos
- El conjunto de datos presenta un fuerte desbalance: ~94% transacciones legítimas, ~6% fraudulentas
- Las direcciones fraudulentas muestran patrones de actividad distintivos, particularmente en frecuencia y volumen de transacciones

### Correlaciones Significativas
- Alta correlación positiva (0.78) entre el número de contratos creados y la probabilidad de fraude
- Correlación negativa (-0.65) entre el tiempo promedio entre transacciones y la probabilidad de fraude
- Las direcciones fraudulentas tienden a interactuar con un conjunto más diverso de otras direcciones

### Variables Más Predictivas
1. Time_Diff_between_first_and_last(Mins)
2. Unique_Received_From_Addresses
3. Unique_Sent_To_Addresses
4. Ratio_Received_Sent
5. Total_Transactions (Sent + Received)

### Tratamiento de Datos
- Se implementaron técnicas SMOTE y SMOTE-Tomek para el balanceo de clases
- Se eliminaron variables con alta colinealidad (>0.85)
- Transformación logarítmica aplicada a variables con distribución sesgada
- Normalización Z-score para variables numéricas

## Estado Actual del Análisis
Este análisis exploratorio ha sido fundamental para entender los patrones de fraude y ha guiado el desarrollo de características y selección de modelos. Los insights obtenidos se han incorporado en todos los modelos implementados.

## Última Actualización
- Fecha: Mayo 2025
- Estado: Completado
- Mantenimiento: Documentación actualizada periódicamente
