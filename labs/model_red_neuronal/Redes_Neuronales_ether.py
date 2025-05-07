# Importación de librerías necesarias
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, metrics, model_selection
import time
from matplotlib import pylab as plt

# Importación de funciones personalizadas
from Funciones import evaluar, evaluarDerivada
from sklearn.neural_network import MLPClassifier

# Cargar el dataset
numeric_df3 = pd.read_csv("transaction_dataset_clean.csv")

# Separar en variables de entrada y objetivo
X = numeric_df3.drop("FLAG", axis=1)  # Eliminar la columna FLAG
y = numeric_df3["FLAG"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

# Definición de los parámetros de la red neuronal
entradas = X.shape[1]
ocultas = 20
salidas = len(np.unique(y))

print(f"Neuronas de entrada = {entradas} ; Neuronas de salida = {salidas}")

# Normalización de datos de entrada
normalizarEntrada = True
if normalizarEntrada:
    standard_scaler = preprocessing.StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

# Transformación de la salida a formato one-hot encoding
y_trainB = np.zeros((len(y_train), salidas))
for o in range(len(y_train)):
    y_trainB[o, y_train.iloc[o]] = 1

# Inicialización de pesos y bias
W1 = np.random.uniform(-1, 1, [ocultas, entradas])
b1 = np.random.uniform(-1, 1, [ocultas, 1])
W2 = np.random.uniform(-1, 1, [salidas, ocultas])
b2 = np.random.uniform(-1, 1, [salidas, 1])

print(f"Total de conexiones en la red: {entradas * ocultas + ocultas * salidas}")

# Funciones de activación
FunH = "sigmoid"
FunO = "tanh"

# Ajuste de salida si la función de activación es tanh
if FunO == "tanh":
    y_trainB = 2 * y_trainB - 1  # Convertir valores de [0,1] a [-1,1]

# Configuración de entrenamiento
alfa = 0.1  # Tasa de aprendizaje
CotaError = 1.0e-4  # Umbral mínimo de error
MAX_ITERA = 400  # Iteraciones máximas
ite = 0
errorAnt = 0
AVGError = 1
errores = []

# Entrenamiento de la red neuronal
while abs(AVGError - errorAnt) > CotaError and ite < MAX_ITERA:
    errorAnt = AVGError
    AVGError = 0
    for e in range(len(X_train)):  # Actualización de la red por cada ejemplo
        xi = X_train[e:e+1, :]
        yi = y_trainB[e:e+1, :]

        # Propagación hacia adelante
        netasH = W1 @ xi.T + b1
        salidasH = evaluar(FunH, netasH)
        netasO = W2 @ salidasH + b2
        salidasO = evaluar(FunO, netasO)

        # Cálculo del error y ajuste de pesos
        ErrorSalida = yi.T - salidasO
        deltaO = ErrorSalida * evaluarDerivada(FunO, salidasO)
        deltaH = evaluarDerivada(FunH, salidasH) * (W2.T @ deltaO)

        W1 += alfa * deltaH @ xi
        b1 += alfa * deltaH
        W2 += alfa * deltaO @ salidasH.T
        b2 += alfa * deltaO

        AVGError += np.mean(ErrorSalida**2)

    # Registro del error
    AVGError /= len(X_train)
    errores.append(AVGError)
    ite += 1

# Graficar evolución del error
plt.plot(range(1, len(errores) + 1), errores, marker="o")
plt.xlabel("Iteraciones")
plt.ylabel("ECM")
plt.grid(True)
plt.show()

print("Error mínimo alcanzado:", min(errores))

# Evaluación en datos de prueba
NetasH = W1 @ X_test.T + b1
SalidasH = evaluar(FunH, NetasH)
NetasO = W2 @ SalidasH + b2
SalidasO = evaluar(FunO, NetasO)

y_pred = np.argmax(SalidasO, axis=0)
print(f"Porcentaje de aciertos en X_test: {metrics.accuracy_score(y_test, y_pred):.3f}")

# Reporte de métricas
report = metrics.classification_report(y_test, y_pred)
print("Informe de métricas:\n", report)

MM = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", MM)