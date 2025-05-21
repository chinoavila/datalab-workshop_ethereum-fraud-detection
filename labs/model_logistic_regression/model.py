import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from pylab import rcParams

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

from collections import Counter

def run_model(X_train, X_test, y_train, y_test, balanced=False):
    if balanced:
      model = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    else:
      model = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg")
    model.fit(X_train, y_train)
    return model

def show_results(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

df = pd.read_csv("../../datasets/transaction_dataset_clean.csv")

print(df.shape) # cuantas filas y columnas tiene el dataset

# analizamos las clases en FLAG
count_classes = pd.Series(df['FLAG']).value_counts() 
print(count_classes)
count_classes.plot(kind = 'bar', rot=0)
LABELS = ["Sin fraude", "Con fraude"]
plt.xticks(range(2), LABELS)
plt.title("Frequency by observation number")
plt.xlabel("FLAG")
plt.ylabel("Number of Observations")

# se definen etiquetas y features
x = pd.get_dummies(df.drop('FLAG', axis=1))
y = df['FLAG']

# se unen las features y las etiquetas para eliminar las filas con nulos de manera consistente.
# Así aseguramos que x e y tengan el mismo número de filas después de la eliminación de nulos.
xy = pd.concat([x, y], axis=1)
xy = xy.dropna()

# se separan las features y etiquetas después de la limpieza
x = xy.drop('FLAG', axis=1)
y = xy['FLAG']

# se divide en dataframe en sets de entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

# entrenamiento, ejecución y análisis del modelo
model = run_model(x_train, x_test, y_train, y_test)
pred_y = model.predict(x_test)
show_results(y_test, pred_y)

# estrategia de penalizacion para compensar
model = run_model(x_train, x_test, y_train, y_test, True)
pred_y = model.predict(x_test)
show_results(y_test, pred_y)

# estrategia de subsampling
nm = NearMiss()
print ("Distribución pre under-sampling {}".format(Counter(y_train)))
x_train_nm, y_train_nm = nm.fit_resample(x_train, y_train)
print ("Distribución post under-sampling {}".format(Counter(y_train)))
model = run_model(x_train_nm, x_test, y_train_nm, y_test, False)
pred_y = model.predict(x_test)
show_results(y_test, pred_y)

# estrategia de oversampling
os = RandomOverSampler()
print ("Distribución pre over-sampling {}".format(Counter(y_train)))
x_train_os, y_train_os = os.fit_resample(x_train, y_train)
print ("Distribución post over-sampling {}".format(Counter(y_train)))
model = run_model(x_train_os, x_test, y_train_os, y_test)
pred_y = model.predict(x_test)
show_results(y_test, pred_y)

# estrategia de smote-tomek
st = SMOTETomek()
print ("Distribución pre smote-tomek {}".format(Counter(y_train)))
x_train_st, y_train_st = st.fit_resample(x_train, y_train)
print ("Distribución post smote-tomek {}".format(Counter(y_train)))
model = run_model(x_train_st, x_test, y_train_st, y_test)
pred_y = model.predict(x_test)
show_results(y_test, pred_y)

# estrategia de ensamble de modelos con balanceo
bbc = BalancedBaggingClassifier(random_state=42)
bbc.fit(x_train, y_train)
pred_y = bbc.predict(x_test)
show_results(y_test, pred_y)

# Guardar el modelo final
joblib.dump(bbc, '../../models/logistic_regression_model.joblib')
print("\nModelo guardado exitosamente en el directorio 'models'")