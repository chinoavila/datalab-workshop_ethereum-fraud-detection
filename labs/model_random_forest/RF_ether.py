# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
df_RF = pd.read_csv("transaction_dataset_clean.csv")

# Preparación de los datos
X = df_RF.drop("FLAG", axis=1)  # Variables de entrada
y = df_RF["FLAG"]  # Variable objetivo

# División en entrenamiento (70%), validación (15%) y test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalización de los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Definimos el modelo base
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

# Definimos el espacio de búsqueda de hiperparámetros
param_grid = {
    "n_estimators": np.arange(50, 300, 50),  # Cantidad de árboles
    "max_depth": [None, 5, 10, 20, 30],  # Profundidad máxima
    "min_samples_split": [2, 5, 10],  # Mínimas muestras para dividir
    "min_samples_leaf": [1, 2, 4],  # Mínimas muestras en una hoja
    "max_features": ["sqrt", "log2", None],  # Método para elegir features en cada división
}

# Configuración de búsqueda aleatoria de hiperparámetros
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,  # Número de combinaciones aleatorias a probar
    cv=5,  # Cross-validation con 5 folds
    verbose=2,
    random_state=42,
    n_jobs=-1,
)


random_search.fit(X_val, y_val)

# Selección del mejor modelo con hiperparámetros óptimos
mejores_params = random_search.best_params_
print("Mejores hiperparámetros encontrados:", mejores_params)

rf_opt = RandomForestClassifier(**mejores_params, class_weight="balanced", random_state=42, n_jobs=-1)
rf_opt.fit(X_train, y_train)

# Evaluación del modelo en entrenamiento
y_train_pred = rf_opt.predict(X_train)
print("Matriz de confusión (Entrenamiento):\n", confusion_matrix(y_train, y_train_pred))
print("\nReporte de clasificación (Entrenamiento):\n", classification_report(y_train, y_train_pred))

# Evaluación del modelo en test
y_test_pred = rf_opt.predict(X_test)
print("\nMatriz de confusión (Test):\n", confusion_matrix(y_test, y_test_pred))
print("\nReporte de clasificación (Test):\n", classification_report(y_test, y_test_pred))

# Curva ROC y cálculo del AUC
y_test_proba = rf_opt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Graficar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Azar (AUC = 0.5)")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC - Random Forest Optimizado")
plt.legend(loc="lower right")
plt.grid()
plt.show()