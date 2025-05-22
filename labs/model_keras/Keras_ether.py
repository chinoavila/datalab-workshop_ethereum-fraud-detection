# Importación de librerías necesarias
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import sys
import os

# Agregar el directorio de common_functions al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common_functions'))
from data_utils import load_and_preprocess_data, normalize_features
from eval_utils import evaluate_model, plot_confusion_matrix, plot_class_distribution
from balance_utils import apply_undersampling, apply_oversampling, apply_smote_tomek

def create_keras_model(input_dim, class_weight=None):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "../../datasets/transaction_dataset_clean.csv"
    )

    # Normalizar características
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

    # Visualizar distribución inicial
    plot_class_distribution(y_train)

    # Configurar early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Calcular class_weight para balanceo
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    class_weight = {
        0: n_samples / (n_classes * np.sum(y_train == 0)),
        1: n_samples / (n_classes * np.sum(y_train == 1))
    }

    # 1. Modelo base sin balance
    print("\n1. Evaluación del modelo base:")
    model = create_keras_model(X_train.shape[1])
    history = model.fit(
        X_train_norm, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    y_pred = (model.predict(X_test_norm) > 0.5).astype(int)
    metrics_base = evaluate_model(y_test, y_pred)
    print(metrics_base['classification_report'])
    plot_confusion_matrix(metrics_base['confusion_matrix'])

    # 2. Modelo con balance de clases
    print("\n2. Evaluación del modelo con balance de clases:")
    model_balanced = create_keras_model(X_train.shape[1])
    history_balanced = model_balanced.fit(
        X_train_norm, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0
    )
    y_pred = (model_balanced.predict(X_test_norm) > 0.5).astype(int)
    metrics_balanced = evaluate_model(y_test, y_pred)
    print(metrics_balanced['classification_report'])
    plot_confusion_matrix(metrics_balanced['confusion_matrix'])

    # 3. Modelo con under-sampling
    print("\n3. Evaluación del modelo con under-sampling:")
    X_train_under, y_train_under = apply_undersampling(X_train_norm, y_train)
    model_under = create_keras_model(X_train.shape[1])
    history_under = model_under.fit(
        X_train_under, y_train_under,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    y_pred = (model_under.predict(X_test_norm) > 0.5).astype(int)
    metrics_under = evaluate_model(y_test, y_pred)
    print(metrics_under['classification_report'])
    plot_confusion_matrix(metrics_under['confusion_matrix'])

    # 4. Modelo con over-sampling
    print("\n4. Evaluación del modelo con over-sampling:")
    X_train_over, y_train_over = apply_oversampling(X_train_norm, y_train)
    model_over = create_keras_model(X_train.shape[1])
    history_over = model_over.fit(
        X_train_over, y_train_over,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    y_pred = (model_over.predict(X_test_norm) > 0.5).astype(int)
    metrics_over = evaluate_model(y_test, y_pred)
    print(metrics_over['classification_report'])
    plot_confusion_matrix(metrics_over['confusion_matrix'])

    # 5. Modelo con SMOTE-Tomek
    print("\n5. Evaluación del modelo con SMOTE-Tomek:")
    X_train_st, y_train_st = apply_smote_tomek(X_train_norm, y_train)
    model_st = create_keras_model(X_train.shape[1])
    history_st = model_st.fit(
        X_train_st, y_train_st,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    y_pred = (model_st.predict(X_test_norm) > 0.5).astype(int)
    metrics_st = evaluate_model(y_test, y_pred)
    print(metrics_st['classification_report'])
    plot_confusion_matrix(metrics_st['confusion_matrix'])

    # Seleccionar el mejor modelo (en este caso, usaremos el balanceado)
    # Luego, guardar modelo y scaler
    best_model = model_balanced
    best_model.save('../../models/keras_model.h5')
    joblib.dump(scaler, '../../models/keras_scaler.joblib')
    print("\nModelo y scaler guardados exitosamente en el directorio 'models'")

if __name__ == "__main__":
    main()