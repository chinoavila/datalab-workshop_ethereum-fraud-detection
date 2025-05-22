import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common_functions'))
from data_utils import load_and_preprocess_data, normalize_features
from eval_utils import evaluate_model, plot_confusion_matrix, plot_class_distribution
from balance_utils import apply_undersampling, apply_oversampling, apply_smote_tomek, get_balanced_ensemble

def create_model(balanced=False):
    params = { 'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg' }
    if balanced:
        params['class_weight'] = 'balanced'
    return LogisticRegression(**params)

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

def main():
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "../../datasets/transaction_dataset_clean.csv"
    )
    
    # Normalizar características
    X_train_norm, X_test_norm, _ = normalize_features(X_train, y_train)
    
    # Visualizar distribución inicial
    plot_class_distribution(y_train)
    
    # 1. Modelo base
    print("\n1. Evaluación del modelo base:")
    model = create_model()
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    metrics_base = evaluate_model(y_test, y_pred)
    print(metrics_base['classification_report'])
    plot_confusion_matrix(metrics_base['confusion_matrix'])
    
    # 2. Modelo con balance de clases
    print("\n2. Evaluación del modelo con balance de clases:")
    model_balanced = create_model(balanced=True)
    model_balanced.fit(X_train_norm, y_train)
    y_pred = model_balanced.predict(X_test_norm)
    metrics_balanced = evaluate_model(y_test, y_pred)
    print(metrics_balanced['classification_report'])
    plot_confusion_matrix(metrics_balanced['confusion_matrix'])
    
    # 3. Modelo con under-sampling
    print("\n3. Evaluación del modelo con under-sampling:")
    X_train_under, y_train_under = apply_undersampling(X_train_norm, y_train)
    model_under = create_model()
    model_under.fit(X_train_under, y_train_under)
    y_pred = model_under.predict(X_test_norm)
    metrics_under = evaluate_model(y_test, y_pred)
    print(metrics_under['classification_report'])
    plot_confusion_matrix(metrics_under['confusion_matrix'])
    
    # 4. Modelo con over-sampling
    print("\n4. Evaluación del modelo con over-sampling:")
    X_train_over, y_train_over = apply_oversampling(X_train_norm, y_train)
    model_over = create_model()
    model_over.fit(X_train_over, y_train_over)
    y_pred = model_over.predict(X_test_norm)
    metrics_over = evaluate_model(y_test, y_pred)
    print(metrics_over['classification_report'])
    plot_confusion_matrix(metrics_over['confusion_matrix'])
    
    # 5. Modelo con SMOTE-Tomek
    print("\n5. Evaluación del modelo con SMOTE-Tomek:")
    X_train_st, y_train_st = apply_smote_tomek(X_train_norm, y_train)
    model_st = create_model()
    model_st.fit(X_train_st, y_train_st)
    y_pred = model_st.predict(X_test_norm)
    metrics_st = evaluate_model(y_test, y_pred)
    print(metrics_st['classification_report'])
    plot_confusion_matrix(metrics_st['confusion_matrix'])
    
    # 6. Modelo con ensamble balanceado
    print("\n6. Evaluación del modelo con ensamble balanceado:")
    bbc = get_balanced_ensemble()
    bbc.fit(X_train_norm, y_train)
    y_pred = bbc.predict(X_test_norm)
    metrics_bbc = evaluate_model(y_test, y_pred)
    print(metrics_bbc['classification_report'])
    plot_confusion_matrix(metrics_bbc['confusion_matrix'])

    # Guardar el modelo final
    joblib.dump(bbc, '../../models/logistic_regression_model.joblib')
    print("\nModelo guardado exitosamente en el directorio 'models'")

if __name__ == "__main__":
    main()
