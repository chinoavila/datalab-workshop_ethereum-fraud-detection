# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Cargar el dataset
df_keras = pd.read_csv("../../datasets/transaction_dataset_clean.csv")

# Separar en variables de entrada y objetivo
X = df_keras.drop("FLAG", axis=1)  # Eliminar la columna FLAG
y = df_keras["FLAG"]

# Dividir en entrenamiento y prueba (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalización de los datos de entrada
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# One-hot encoding de la variable objetivo
y_trainB = to_categorical(y_train)
y_testB = to_categorical(y_test)

# Ajuste del rango de salida a [-1, 1] si la función de activación lo requiere
y_trainB = 2 * y_trainB - 1
y_testB = 2 * y_testB - 1

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(20, input_dim=X_train_norm.shape[1], activation="sigmoid"))  # Capa oculta con 20 neuronas
model.add(Dense(y_trainB.shape[1], activation="tanh"))  # Capa de salida con activación tanh

# Compilar el modelo con optimizador SGD y función de pérdida MSE
model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

# Entrenamiento del modelo
history = model.fit(X_train_norm, y_trainB, epochs=100, verbose=1)

# Graficar la evolución del error durante el entrenamiento
plt.plot(history.history["loss"])
plt.xlabel("Época")
plt.ylabel("Pérdida (MSE)")
plt.title("Evolución del error")
plt.grid(True)
plt.show()

# Predicción de clases en conjunto de entrenamiento
y_pred_probs = model.predict(X_train_norm)
y_pred = np.argmax(y_pred_probs, axis=1)

# Cálculo de la precisión en entrenamiento
acc = metrics.accuracy_score(y_train, y_pred)
print(f"Accuracy en entrenamiento: {acc:.3f}")

# Matriz de confusión
MM = confusion_matrix(y_train, y_pred)
print("Matriz de confusión:")
print(MM)

# Detalle de aciertos y errores por clase
print("\nAciertos y errores por clase:")
for i in range(MM.shape[0]):
    aciertos = MM[i, i]
    errores = sum(MM[i, :]) - aciertos
    print(f"Clase {i}: Aciertos = {aciertos}, Errores = {errores}")

# Evaluación del modelo en test
test_loss, test_acc = model.evaluate(X_test_norm, y_testB, verbose=0)
print(f"Accuracy en test: {test_acc:.3f}")

# Predicción en datos de prueba
y_pred_probs = model.predict(X_test_norm)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de confusión y reporte de clasificación
MM = confusion_matrix(y_test, y_pred)
print("Matriz de confusión (test):")
print(MM)

print("\nAciertos y errores por clase (test):")
for i in range(MM.shape[0]):
    aciertos = MM[i, i]
    errores = sum(MM[i, :]) - aciertos
    print(f"Clase {i}: Aciertos = {aciertos}, Errores = {errores}")

# Reporte de clasificación en test
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo y el scaler
joblib.dump(model, '../../models/keras_model.joblib')
joblib.dump(scaler, '../../models/keras_scaler.joblib')
print("\nModelo y scaler guardados exitosamente en el directorio 'models'")