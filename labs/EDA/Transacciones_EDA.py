# Importación de librerías necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("transaction_dataset.csv")

# Información general del dataset
df.info()

# Seleccionar solo las columnas numéricas
numeric_df = df.select_dtypes(include=["number"])

# Rellenar valores NaN con la media de cada columna
numeric_df2 = numeric_df.fillna(numeric_df.mean())

# Verificar que ya no hay valores NaN
print(numeric_df2.isnull().sum())

# Calcular la correlación de FLAG con las demás columnas numéricas
cor = numeric_df2.corr()["FLAG"].sort_values(ascending=False)
print(cor)

# Eliminar espacios extra en los nombres de las columnas
numeric_df2.columns = numeric_df2.columns.str.strip()

# Verificar la existencia de ciertas columnas específicas
columns_to_check = [
    "ERC20 avg time between sent tnx",
    "ERC20 avg time between rec tnx",
    "ERC20 avg time between rec 2 tnx",
    "ERC20 avg time between contract tnx",
    "ERC20 min val sent contract",
    "ERC20 max val sent contract",
    "ERC20 avg val sent contract"
]

# Comprobar número de valores únicos en las columnas específicas
for col in columns_to_check:
    if col in numeric_df2.columns:
        print(f"{col}: {numeric_df2[col].nunique()}")
    else:
        print(f"La columna '{col}' no está en numeric_df.")

# Eliminar columnas con valores repetidos o irrelevantes
numeric_df3 = numeric_df2.drop(columns=columns_to_check)

col_irre = ["Unnamed: 0", "Index"]
numeric_df3 = numeric_df3.drop(columns=col_irre)

# Resumen estadístico de los datos numéricos
print('\nCantidad de columnas numéricas restantes:', len(numeric_df3.columns))
print(numeric_df3.describe())

# Calcular la correlación de FLAG con las demás columnas numéricas
cor2 = numeric_df3.corr()["FLAG"].sort_values(ascending=False)
print(cor2)

# Generar matriz de correlación
cor_matrix = numeric_df3.corr()

# Crear un mapa de calor de la correlación
plt.figure(figsize=(8, 6))
sns.heatmap(cor_matrix, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de calor de la correlación entre variables")
plt.show()

# Obtener pares de correlación
cor_pairs = cor_matrix.unstack()
cor_pairs = cor_pairs[cor_pairs != 1.0].drop_duplicates()

# Filtrar correlaciones fuertes
strong_corr = cor_pairs[(cor_pairs > 0.75) | (cor_pairs < -0.75)]

# Mostrar las correlaciones más fuertes
print("Los pares con mayor correlación son:")
print(strong_corr.sort_values(ascending=False))

# Columnas redundantes por alta correlación
col_redun = [
    "max val sent to contract",
    "ERC20 total Ether received",
    "ERC20 max val sent",
    "ERC20 min val sent",
    "ERC20 total ether sent",
    "ERC20 uniq rec token name"
]

numeric_df3 = numeric_df3.drop(columns=col_redun)

# Resumen tras eliminar columnas redundantes
print('\nCantidad de columnas numéricas finales:', len(numeric_df3.columns))
print(numeric_df3.describe())

# Conteo y porcentaje de cada clase en FLAG
conteo = numeric_df3["FLAG"].value_counts()
porcentaje = numeric_df3["FLAG"].value_counts(normalize=True) * 100

print("Conteo de FLAG:")
print(conteo)
print("\nPorcentaje de FLAG:")
print(porcentaje)

# Histograma de la variable objetivo FLAG
plt.hist(numeric_df3["FLAG"], bins=50, color="blue", alpha=0.7)
plt.xlabel("Es no fraude o fraude")
plt.ylabel("Frecuencia")
plt.title("Histograma de FLAG")
plt.grid(True)
plt.show()

# Guardar el dataset limpio
numeric_df3.to_csv("transaction_dataset_clean.csv", index=False)
print("Archivo transaction_dataset_clean.csv guardado correctamente.")