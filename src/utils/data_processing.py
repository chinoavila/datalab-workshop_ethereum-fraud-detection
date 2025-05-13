import pandas as pd
import base64
from sklearn.metrics import classification_report

def get_model_metrics(report):
    metrics = {}
    metrics['Precisión (Fraude)'] = report['1']['precision']
    metrics['Recall (Fraude)'] = report['1']['recall']
    metrics['F1-Score (Fraude)'] = report['1']['f1-score']
    metrics['Precisión (No Fraude)'] = report['0']['precision']
    metrics['Recall (No Fraude)'] = report['0']['recall']
    metrics['F1-Score (No Fraude)'] = report['0']['f1-score']
    metrics['Exactitud Global'] = report['accuracy']
    return metrics

def get_classification_report(y_test, pred_y):
    return classification_report(y_test, pred_y, output_dict=True)

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="fraud_predictions.csv">Descargar CSV con predicciones</a>'
