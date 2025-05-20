import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

from xgboost import XGBClassifier

# === Función para entrenar el modelo ===
def run_xgb_model(X_train, y_train, scale_weight=None):
    if scale_weight:
        ratio = scale_weight
    else:
        ratio = 1  # sin ajuste
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio, random_state=42)
    model.fit(X_train, y_train)
    return model

# === Función para mostrar resultados ===
def show_results(y_test, pred_y, title):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test, pred_y, target_names=LABELS))

# === Carga y preprocesamiento de datos ===
df = pd.read_csv("../../datasets/transaction_dataset.csv")

print(df.shape)  # filas y columnas
count_classes = pd.Series(df['FLAG']).value_counts()
print(count_classes)

LABELS = ["Sin fraude", "Con fraude"]
count_classes.plot(kind='bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title("Frecuencia por clase")
plt.xlabel("FLAG")
plt.ylabel("Observaciones")
plt.show()

# Preprocesamiento
# Eliminar columnas irrelevantes o identificadoras
columns_to_drop = ["Unnamed: 0", "Index"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
# Filtrar solo columnas numéricas
df_numeric = df.select_dtypes(include=["number"])
# Imputar valores nulos con la media
df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))
# Verificamos que FLAG está en el dataset
assert 'FLAG' in df_numeric.columns, "La columna 'FLAG' no está presente en el dataset numérico."
# Separar variables predictoras (X) y la variable objetivo (y)
x = df_numeric.drop("FLAG", axis=1)
y = df_numeric["FLAG"]
# División del dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42, stratify=y)

# === 1. Sin balanceo (modelo base) ===
model = run_xgb_model(x_train, y_train)
pred_y = model.predict(x_test)
show_results(y_test, pred_y, "Sin Balanceo")

# === 2. Con scale_pos_weight ===
ratio = Counter(y_train)[0] / Counter(y_train)[1]
model = run_xgb_model(x_train, y_train, scale_weight=ratio)
pred_y = model.predict(x_test)
show_results(y_test, pred_y, "scale_pos_weight")

# === 3. Under-sampling (NearMiss) ===
print("Under-sampling:")
print("Antes:", Counter(y_train))
x_train_nm, y_train_nm = NearMiss().fit_resample(x_train, y_train)
print("Después:", Counter(y_train_nm))
model = run_xgb_model(x_train_nm, y_train_nm)
pred_y = model.predict(x_test)
show_results(y_test, pred_y, "Under-sampling")

# === 4. Over-sampling ===
print("Over-sampling:")
print("Antes:", Counter(y_train))
x_train_os, y_train_os = RandomOverSampler().fit_resample(x_train, y_train)
print("Después:", Counter(y_train_os))
model = run_xgb_model(x_train_os, y_train_os)
pred_y = model.predict(x_test)
show_results(y_test, pred_y, "Over-sampling")

# === 5. SMOTE-Tomek ===
print("SMOTE-Tomek:")
print("Antes:", Counter(y_train))
x_train_st, y_train_st = SMOTETomek().fit_resample(x_train, y_train)
print("Después:", Counter(y_train_st))
model = run_xgb_model(x_train_st, y_train_st)
pred_y = model.predict(x_test)
show_results(y_test, pred_y, "SMOTE-Tomek")

# === 6. Ensamble balanceado ===
bbc = BalancedBaggingClassifier(base_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                                random_state=42, n_estimators=10)
bbc.fit(x_train, y_train)
pred_y = bbc.predict(x_test)
show_results(y_test, pred_y, "Balanced Bagging Ensemble")
