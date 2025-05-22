import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, labels=None):
    if labels is None:
        labels = ["Sin fraude", "Con fraude"]
        
    # Calcular métricas básicas
    metrics_dict = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'classification_report': metrics.classification_report(y_true, y_pred),
        'confusion_matrix': metrics.confusion_matrix(y_true, y_pred)
    }
    
    return metrics_dict

def plot_confusion_matrix(confusion_matrix, labels=None):
    if labels is None:
        labels = ["Sin fraude", "Con fraude"]
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()

def plot_class_distribution(y, labels=None):
    if labels is None:
        labels = ["Sin fraude", "Con fraude"]
        
    class_counts = pd.Series(y).value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar')
    plt.title("Distribución de Clases")
    plt.xlabel("Clase")
    plt.ylabel("Número de Observaciones")
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.show()
