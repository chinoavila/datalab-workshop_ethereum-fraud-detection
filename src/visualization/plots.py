import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, pred_y, labels):
    conf_matrix = confusion_matrix(y_test, pred_y)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", ax=ax)
    plt.title("Matriz de Confusión")
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    return fig

def plot_class_distribution(df, column='FLAG'):
    fig, ax = plt.subplots(figsize=(8, 6))
    count_classes = pd.Series(df[column]).value_counts()
    count_classes.plot(kind='bar', rot=0, ax=ax)
    plt.xticks(range(2), ["Sin fraude", "Con fraude"])
    plt.title("Distribución de Clases")
    plt.xlabel("FLAG")
    plt.ylabel("Número de Observaciones")
    return fig
