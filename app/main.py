import streamlit as st
import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import run_model, apply_balancing
from src.utils.data_processing import get_model_metrics, get_classification_report, get_download_link
from src.visualization.plots import plot_confusion_matrix, plot_class_distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

from collections import Counter

# Configuración de la página
st.set_page_config(
    page_title="Detección de Fraude",
    page_icon="🔍",
    layout="wide"
)

# Título principal
st.title('🔍 Sistema de Detección de Fraude')
st.markdown('---')

# Sidebar con instrucciones y opciones
with st.sidebar:
    st.header('Configuración')
    
    st.markdown('### Instrucciones')
    st.info("""
    1. Sube tu archivo CSV con datos de transacciones
    2. Selecciona la estrategia de balanceo
    3. Configura los parámetros del modelo
    4. Visualiza los resultados
    """)
    
    test_size = st.slider('Porcentaje para test (%)', 10, 90, 30)
    test_size = test_size / 100

    st.markdown('### Estrategia de Balanceo')
    balancing_strategy = st.selectbox(
        'Selecciona estrategia de balanceo',
        ('Sin balanceo', 'Class Weight', 'Under-sampling (NearMiss)', 
         'Over-sampling', 'SMOTE-Tomek', 'Balanced Bagging Classifier')
    )

# Carga de archivos
st.header('1. Carga de Datos')
uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de transacciones", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f'Archivo cargado exitosamente: {uploaded_file.name}')
        
        # Mostrar información del dataset
        st.header('2. Información del Dataset')
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
            st.write("Primeras 5 filas:")
            st.dataframe(df.head())
            
        with col2:
            if 'FLAG' in df.columns:
                st.write("Distribución de clases:")
                count_classes = pd.Series(df['FLAG']).value_counts()
                st.write(count_classes)
                st.pyplot(plot_class_distribution(df))
            else:
                st.error("El dataset debe contener una columna llamada 'FLAG' con las etiquetas.")
                st.stop()
        
        # Preprocesamiento
        st.header('3. Preprocesamiento')
        
        with st.spinner('Procesando datos...'):
            # Definir features y target
            if 'FLAG' in df.columns:
                x = pd.get_dummies(df.drop('FLAG', axis=1))
                y = df['FLAG']
                
                # Unir y eliminar filas con nulos
                xy = pd.concat([x, y], axis=1)
                nulos_antes = xy.isna().sum().sum()
                xy = xy.dropna()
                nulos_despues = xy.isna().sum().sum()
                
                # Separar features y target después de la limpieza
                x = xy.drop('FLAG', axis=1)
                y = xy['FLAG']
                
                st.write(f"Valores nulos encontrados: {nulos_antes}")
                st.write(f"Valores nulos después de la limpieza: {nulos_despues}")
                st.write(f"Dimensiones después de la limpieza: {x.shape[0]} filas x {x.shape[1]} columnas")
                
                # División train-test
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
                st.write(f"Datos de entrenamiento: {x_train.shape[0]} muestras")
                st.write(f"Datos de prueba: {x_test.shape[0]} muestras")
                
                # Entrenamiento y evaluación de modelos
                st.header('4. Entrenamiento del Modelo')
                
                if st.button('Entrenar Modelo'):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Aplicar la estrategia de balanceo seleccionada
                    status_text.text("Aplicando estrategia de balanceo...")
                    progress_bar.progress(10)
                    
                    if balancing_strategy == 'Sin balanceo':
                        model = run_model(x_train, x_test, y_train, y_test, False)
                        x_train_balanced, y_train_balanced = x_train, y_train
                        
                    elif balancing_strategy == 'Class Weight':
                        model = run_model(x_train, x_test, y_train, y_test, True)
                        x_train_balanced, y_train_balanced = x_train, y_train
                        
                    elif balancing_strategy == 'Under-sampling (NearMiss)':
                        status_text.text("Aplicando Under-sampling...")
                        nm = NearMiss()
                        x_train_balanced, y_train_balanced = nm.fit_resample(x_train, y_train)
                        model = run_model(x_train_balanced, x_test, y_train_balanced, y_test)
                        
                    elif balancing_strategy == 'Over-sampling':
                        status_text.text("Aplicando Over-sampling...")
                        os = RandomOverSampler()
                        x_train_balanced, y_train_balanced = os.fit_resample(x_train, y_train)
                        model = run_model(x_train_balanced, x_test, y_train_balanced, y_test)
                        
                    elif balancing_strategy == 'SMOTE-Tomek':
                        status_text.text("Aplicando SMOTE-Tomek...")
                        st = SMOTETomek()
                        x_train_balanced, y_train_balanced = st.fit_resample(x_train, y_train)
                        model = run_model(x_train_balanced, x_test, y_train_balanced, y_test)
                        
                    elif balancing_strategy == 'Balanced Bagging Classifier':
                        status_text.text("Aplicando Balanced Bagging Classifier...")
                        model = BalancedBaggingClassifier(random_state=42)
                        model.fit(x_train, y_train)
                    
                    progress_bar.progress(50)
                    status_text.text("Realizando predicciones...")
                    
                    # Distribución después del balanceo
                    st.subheader("Distribución después del balanceo")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    pd.Series(Counter(y_train_balanced)).plot(kind='bar', rot=0, ax=ax)
                    plt.xticks(range(2), ["Sin fraude", "Con fraude"])
                    plt.title("Distribución después del balanceo")
                    st.pyplot(fig)
                    
                    # Realizar predicciones
                    pred_y = model.predict(x_test)
                    progress_bar.progress(80)
                    status_text.text("Generando métricas y visualizaciones...")
                    
                    # Mostrar resultados
                    st.header('5. Resultados')
                    
                    # Métricas generales
                    report = get_classification_report(y_test, pred_y)
                    metrics = get_model_metrics(report)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Matriz de Confusión")
                        LABELS = ["Sin fraude", "Con fraude"]
                        st.pyplot(plot_confusion_matrix(y_test, pred_y, LABELS))
                    
                    with col2:
                        st.subheader("Métricas de Rendimiento")
                        for metric, value in metrics.items():
                            st.metric(metric, f"{value:.4f}")
                    
                    # Predicciones en los datos de prueba
                    st.subheader("Predicciones en datos de prueba")
                    results_df = pd.DataFrame({
                        'Real': y_test,
                        'Predicción': pred_y,
                        'Correcto': y_test == pred_y
                    }).reset_index()
                    
                    st.dataframe(results_df)
                    st.markdown(get_download_link(results_df), unsafe_allow_html=True)
                    
                    progress_bar.progress(100)
                    status_text.text("¡Proceso completado!")
            else:
                st.error("El dataset debe contener una columna llamada 'FLAG' con las etiquetas.")
                
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("Por favor sube un archivo CSV para comenzar.")

# Footer
st.markdown('---')
st.markdown('Desarrollado para la detección de fraude en transacciones financieras')