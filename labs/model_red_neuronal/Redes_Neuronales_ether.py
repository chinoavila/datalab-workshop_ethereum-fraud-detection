# Importación de librerías necesarias
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Agregar el directorio de common_functions al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common_functions'))
from eval_functions import evaluar, evaluarDerivada
from data_utils import load_and_preprocess_data, normalize_features
from eval_utils import evaluate_model, plot_confusion_matrix, plot_class_distribution

def create_neural_network(input_size, hidden_size=20, output_size=2):
    W1 = np.random.uniform(-1, 1, [hidden_size, input_size])
    b1 = np.random.uniform(-1, 1, [hidden_size, 1])
    W2 = np.random.uniform(-1, 1, [output_size, hidden_size])
    b2 = np.random.uniform(-1, 1, [output_size, 1])
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def train_neural_network(X_train, y_train, params, config):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    errores = []
    ite = 0
    errorAnt = 0
    AVGError = 1
    # One-hot encoding de las etiquetas
    y_trainB = np.zeros((len(y_train), config['output_size']))
    for o in range(len(y_train)):
        y_trainB[o, y_train.iloc[o]] = 1
    # Ajuste para función tanh
    if config['FunO'] == 'tanh':
        y_trainB = 2 * y_trainB - 1
    while abs(AVGError - errorAnt) > config['CotaError'] and ite < config['MAX_ITERA']:
        errorAnt = AVGError
        AVGError = 0  
        for e in range(len(X_train)):
            xi = X_train[e:e+1, :]
            yi = y_trainB[e:e+1, :]
            # Forward propagation
            netasH = W1 @ xi.T + b1
            salidasH = evaluar(config['FunH'], netasH)
            netasO = W2 @ salidasH + b2
            salidasO = evaluar(config['FunO'], netasO)
            # Backpropagation
            ErrorSalida = yi.T - salidasO
            deltaO = ErrorSalida * evaluarDerivada(config['FunO'], salidasO)
            deltaH = evaluarDerivada(config['FunH'], salidasH) * (W2.T @ deltaO)
            # Actualización de pesos
            W1 += config['alfa'] * deltaH @ xi
            b1 += config['alfa'] * deltaH
            W2 += config['alfa'] * deltaO @ salidasH.T
            b2 += config['alfa'] * deltaO
            AVGError += np.mean(ErrorSalida**2)
        AVGError /= len(X_train)
        errores.append(AVGError)
        ite += 1
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, errores

def predict(X, params, config):
    NetasH = params['W1'] @ X.T + params['b1']
    SalidasH = evaluar(config['FunH'], NetasH)
    NetasO = params['W2'] @ SalidasH + params['b2']
    SalidasO = evaluar(config['FunO'], NetasO)
    return np.argmax(SalidasO, axis=0)

def main():
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "../../datasets/transaction_dataset_clean.csv"
    )

    # Normalizar características
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

    # Configuración de la red neuronal
    config = {
        'input_size': X_train.shape[1],
        'hidden_size': 20,
        'output_size': len(np.unique(y_train)),
        'alfa': 0.1,
        'CotaError': 1.0e-4,
        'MAX_ITERA': 400,
        'FunH': 'sigmoid',
        'FunO': 'tanh'
    }
    print(f"Arquitectura de la red: {config['input_size']}-{config['hidden_size']}-{config['output_size']}")

    # Inicializar red neuronal
    params = create_neural_network(
        config['input_size'], 
        config['hidden_size'],
        config['output_size']
    )

    # Entrenar red neuronal
    params, errores = train_neural_network(X_train_norm, y_train, params, config)

    # Graficar evolución del error
    plt.plot(range(1, len(errores) + 1), errores, marker="o")
    plt.xlabel("Iteraciones")
    plt.ylabel("ECM")
    plt.grid(True)
    plt.show()
    print("Error mínimo alcanzado:", min(errores))

    # Evaluar modelo
    y_pred = predict(X_test_norm, params, config)
    metrics_dict = evaluate_model(y_test, y_pred)
    print("\nInforme de métricas:")
    print(metrics_dict['classification_report'])
    plot_confusion_matrix(metrics_dict['confusion_matrix'])

    # Guardar modelo y scaler
    model_data = { 'params': params, 'config': config }
    joblib.dump(model_data, '../../models/red_neuronal_model.joblib')
    joblib.dump(scaler, '../../models/red_neuronal_scaler.joblib')
    print("\nModelo y scaler guardados exitosamente en el directorio 'models'")

if __name__ == "__main__":
    main()