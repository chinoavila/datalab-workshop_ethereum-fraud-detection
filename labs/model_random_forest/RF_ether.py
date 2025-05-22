# Importación de librerías necesarias
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
import os

# Agregar el directorio de common_functions al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common_functions'))
from data_utils import load_and_preprocess_data, normalize_features
from eval_utils import evaluate_model, plot_confusion_matrix, plot_class_distribution
from balance_utils import apply_undersampling, apply_oversampling, apply_smote_tomek, get_balanced_ensemble

def create_random_forest(balanced=False, n_estimators=100):
    params = {
        'n_estimators': n_estimators,
        'random_state': 42,
        'n_jobs': -1
    }
    if balanced:
        params['class_weight'] = 'balanced'
    return RandomForestClassifier(**params)

def main():
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "../../datasets/transaction_dataset_clean.csv"
    )

    # Normalizar características
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

    # Visualizar distribución inicial
    plot_class_distribution(y_train)

    # 1. Modelo base
    print("\n1. Evaluación del modelo base:")
    model = create_random_forest()
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    metrics_base = evaluate_model(y_test, y_pred)
    print(metrics_base['classification_report'])
    plot_confusion_matrix(metrics_base['confusion_matrix'])

    # 2. Modelo con balance de clases
    print("\n2. Evaluación del modelo con balance de clases:")
    model_balanced = create_random_forest(balanced=True)
    model_balanced.fit(X_train_norm, y_train)
    y_pred = model_balanced.predict(X_test_norm)
    metrics_balanced = evaluate_model(y_test, y_pred)
    print(metrics_balanced['classification_report'])
    plot_confusion_matrix(metrics_balanced['confusion_matrix'])

    # 3. Modelo con under-sampling
    print("\n3. Evaluación del modelo con under-sampling:")
    X_train_under, y_train_under = apply_undersampling(X_train_norm, y_train)
    model_under = create_random_forest()
    model_under.fit(X_train_under, y_train_under)
    y_pred = model_under.predict(X_test_norm)
    metrics_under = evaluate_model(y_test, y_pred)
    print(metrics_under['classification_report'])
    plot_confusion_matrix(metrics_under['confusion_matrix'])

    # 4. Modelo con over-sampling
    print("\n4. Evaluación del modelo con over-sampling:")
    X_train_over, y_train_over = apply_oversampling(X_train_norm, y_train)
    model_over = create_random_forest()
    model_over.fit(X_train_over, y_train_over)
    y_pred = model_over.predict(X_test_norm)
    metrics_over = evaluate_model(y_test, y_pred)
    print(metrics_over['classification_report'])
    plot_confusion_matrix(metrics_over['confusion_matrix'])

    # 5. Modelo con SMOTE-Tomek
    print("\n5. Evaluación del modelo con SMOTE-Tomek:")
    X_train_st, y_train_st = apply_smote_tomek(X_train_norm, y_train)
    model_st = create_random_forest()
    model_st.fit(X_train_st, y_train_st)
    y_pred = model_st.predict(X_test_norm)
    metrics_st = evaluate_model(y_test, y_pred)
    print(metrics_st['classification_report'])
    plot_confusion_matrix(metrics_st['confusion_matrix'])

    # Seleccionar el mejor modelo (en este caso, usaremos el balanceado)
    # Luego, guardar modelo y scaler
    best_model = model_balanced
    joblib.dump(best_model, '../../models/random_forest_model.joblib')
    joblib.dump(scaler, '../../models/random_forest_scaler.joblib')
    print("\nModelo y scaler guardados exitosamente en el directorio 'models'")

if __name__ == "__main__":
    main()