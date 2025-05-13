import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter

# Inicialización del estado de la sesión
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'test_size' not in st.session_state:
    st.session_state.test_size = 0.3
if 'balancing_strategy' not in st.session_state:
    st.session_state.balancing_strategy = 'Sin balanceo'

# Configuración de la página
st.set_page_config(page_title="Detección de Fraude", page_icon="🔍", layout="wide")

# Barra de progreso
st.progress(st.session_state.current_step / 4)

# Título y descripción
st.title('🔍 Sistema de Detección de Fraude')
st.markdown(f"**Paso {st.session_state.current_step} de 4**")
st.markdown('---')

def next_step():
    st.session_state.current_step += 1

def prev_step():
    st.session_state.current_step -= 1

# Sidebar
with st.sidebar:
    st.header('Navegación')
    st.write(f'Paso actual: {st.session_state.current_step}')
    st.markdown('''
    1. Carga de datos
    2. Configuración del modelo
    3. Entrenamiento
    4. Resultados
    ''')

# Funciones de utilidad
def run_model(X_train, X_test, y_train, y_test, balanced=False):
    if balanced:
        model = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg", class_weight="balanced")
    else:
        model = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg")
    model.fit(X_train, y_train)
    return model

def get_classification_report(y_test, pred_y):
    return classification_report(y_test, pred_y, output_dict=True)

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

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="fraud_predictions.csv">Descargar CSV con predicciones</a>'
    return href

# Contenido según el paso actual
if st.session_state.current_step == 1:
    st.header('1. Carga de Datos')
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de transacciones", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success(f'Archivo cargado exitosamente: {uploaded_file.name}')
            st.dataframe(df.head())
            
            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                if st.button('Siguiente →', on_click=next_step):
                    pass
                    
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")

elif st.session_state.current_step == 2:
    st.header('2. Configuración del Modelo')
    
    if st.session_state.data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.test_size = st.slider('Porcentaje para test (%)', 10, 90, 30) / 100
            st.session_state.balancing_strategy = st.selectbox(
                'Estrategia de balanceo',
                ('Sin balanceo', 'Class Weight', 'Under-sampling', 'Over-sampling', 'SMOTE-Tomek')
            )
            
        # Mostrar distribución actual de clases
        if 'FLAG' in st.session_state.data.columns:
            with col2:
                st.pyplot(plot_class_distribution(st.session_state.data))
            
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button('← Anterior', on_click=prev_step):
                pass
        with col2:
            if st.button('Siguiente →', on_click=next_step):
                pass
    else:
        st.warning("Por favor, carga los datos primero")
        if st.button('← Volver a carga de datos', on_click=prev_step):
            pass

elif st.session_state.current_step == 3:
    st.header('3. Entrenamiento')
    if st.session_state.data is not None:
        try:
            df = st.session_state.data
            
            if 'FLAG' not in df.columns:
                st.error("El dataset debe contener una columna llamada 'FLAG'")
                st.stop()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if st.button('Entrenar Modelo'):
                with st.spinner('Preparando datos...'):
                    # Preparación de datos
                    x = pd.get_dummies(df.drop('FLAG', axis=1))
                    y = df['FLAG']
                    
                    # Limpieza de nulos
                    xy = pd.concat([x, y], axis=1)
                    nulos_antes = xy.isna().sum().sum()
                    xy = xy.dropna()
                    x = xy.drop('FLAG', axis=1)
                    y = xy['FLAG']
                    
                    progress_bar.progress(25)
                    status_text.text("Dividiendo datos...")
                    
                    # División de datos
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, 
                        test_size=st.session_state.test_size, 
                        random_state=42
                    )
                    
                    progress_bar.progress(50)
                    status_text.text("Entrenando modelo...")
                    
                    # Entrenamiento
                    model = run_model(x_train, x_test, y_train, y_test, 
                                   st.session_state.balancing_strategy == 'Class Weight')
                    
                    progress_bar.progress(75)
                    status_text.text("Generando predicciones...")
                    
                    # Predicciones
                    pred_y = model.predict(x_test)
                    
                    # Guardar resultados
                    st.session_state.model_results = {
                        'predictions': pred_y,
                        'y_test': y_test,
                        'report': get_classification_report(y_test, pred_y)
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("¡Proceso completado!")
                    next_step()
                    
        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
            
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button('← Anterior', on_click=prev_step):
                pass

elif st.session_state.current_step == 4:
    st.header('4. Resultados')
    
    if st.session_state.model_results is not None:
        results = st.session_state.model_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matriz de Confusión")
            LABELS = ["Sin fraude", "Con fraude"]
            st.pyplot(plot_confusion_matrix(
                results['y_test'], 
                results['predictions'], 
                LABELS
            ))
        
        with col2:
            st.subheader("Métricas de Rendimiento")
            metrics = get_model_metrics(results['report'])
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")
        
        # Botón para descargar resultados
        results_df = pd.DataFrame({
            'Real': results['y_test'],
            'Predicción': results['predictions'],
            'Correcto': results['y_test'] == results['predictions']
        }).reset_index()
        
        st.dataframe(results_df)
        st.markdown(get_download_link(results_df), unsafe_allow_html=True)
    else:
        st.warning("No hay resultados disponibles. Por favor, entrena el modelo primero.")
    
    col1, _ = st.columns([1, 3])
    with col1:
        if st.button('← Volver al inicio', on_click=lambda: setattr(st.session_state, 'current_step', 1)):
            pass

# Footer
st.markdown('---')
st.markdown('Desarrollado para la detección de fraude en transacciones financieras')