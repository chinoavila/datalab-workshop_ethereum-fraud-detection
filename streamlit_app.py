import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob
import os
import sys
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Agregar directorios de funciones al path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'common_functions'))
sys.path.append(os.path.join(project_root, 'scripts', 'get_data'))
sys.path.append(os.path.join(project_root, 'scripts', 'model_evaluation'))

# Importar funciones del script original
from scripts.model_evaluation.evaluate_models import (
    cargar_modelo,
    cargar_scaler,
    asegurar_columnas,
    predecir_modelo
)

# Importar funciones para descarga de datos
from scripts.get_data.generate_eth_features_history import (
    get_recent_transfers as get_recent_transactions,
    get_transaction_by_hash as get_transaction_data,
    extract_features as generate_features,
    get_historical_transfers_for_address
)

def cargar_dataset_mas_reciente():
    """Carga el dataset más reciente o genera uno nuevo si no existe"""
    features_dir = os.path.join(project_root, "features_downloads")
    initial_file = os.path.join(features_dir, "features_recent_*_*.csv")
    archivos = glob.glob(initial_file)
    if not archivos:
        try:
            st.warning("No se encontraron datos recientes. Generando nuevo dataset...")
            # Configuración por defecto para generar nuevos datos
            minutes = 1  # último minuto
            max_tx = 5  # máximo 10 transacciones            
            get_data_path = os.path.join(project_root, "scripts", "get_data")
            comando = [sys.executable, "generate_eth_features_history.py", 
                                                "--minutes", str(minutes), 
                                                "--max_tx", str(max_tx)]
            os.chdir(get_data_path) # nos movemos al directorio get_data para ejecutar el comando
            process = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8" # <- Esto fuerza la decodificación como UTF-8
            )
            output_placeholder = st.empty()
            for line in process.stdout:
                output_placeholder.info(line.strip()) # Solo muestra la última línea
            process.wait()
            os.chdir(project_root) # volvemos a la raiz del proyecto

            if process.returncode == 0:
                st.success("Script ejecutado correctamente.")
            else:
                st.error("Error al generar nuevos datos")
                return None 
        except Exception as e:
            st.error(f'No se pudo ejecutar el script de generación de nuevos datos: {e}')
            return None
    # Buscar el archivo generado
    archivos = glob.glob(initial_file)
    if not archivos:
        st.error("No se generó el archivo de features")
        return None
    archivo_mas_reciente = max(archivos, key=os.path.getmtime)
    st.info(f"Dataset cargado: {archivo_mas_reciente}")
    return pd.read_csv(archivo_mas_reciente)

def mostrar_metricas(y_true, y_pred, y_prob):
    """Muestra las métricas de evaluación"""
    metricas = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred)
    }
    
    # Crear DataFrame con métricas
    df_metricas = pd.DataFrame({
        'Métrica': list(metricas.keys()),
        'Valor': [f"{v:.3f}" for v in metricas.values()]
    })
    
    return df_metricas

def descargar_datos_recientes(minutes, max_tx, output_placeholder):
    """Descarga datos de transacciones recientes y su histórico"""
    with st.spinner("Descargando transacciones recientes..."):
        try:
            # Obtener transacciones recientes
            recent_tx = get_recent_transactions(minutes=minutes, max_tx=max_tx)
            if not recent_tx:
                st.error("No se encontraron transacciones recientes")
                return None
            
            output_placeholder.info(f"Se obtuvieron {len(recent_tx)} transacciones recientes")
            
            # Obtener direcciones únicas de remitentes
            addresses = set()
            for tx in recent_tx:
                if tx.get("from"):
                    addresses.add(tx["from"])
            
            output_placeholder.info(f"Consultando histórico para {len(addresses)} direcciones...")
            
            # Obtener histórico para cada dirección
            all_hist_txs = recent_tx.copy()
            for i, addr in enumerate(addresses):
                output_placeholder.info(f"Consultando histórico para {addr} ({i+1}/{len(addresses)})")
                hist_txs = get_historical_transfers_for_address(addr, max_tx=max_tx)
                all_hist_txs.extend(hist_txs)
                time.sleep(0.25)  # Evitar rate limiting
            
            output_placeholder.info(f"Total de transacciones históricas recopiladas: {len(all_hist_txs)}")
            
            # Generar features
            output_placeholder.info("Extrayendo features...")
            features = generate_features(all_hist_txs, recent_tx)
            
            # Guardar resultados
            timestamp = time.strftime("%Y_%m_%d_%H_%M")
            filename = f"features_recent_{minutes}m_{max_tx}tx_{timestamp}.csv"
            file_path = os.path.join(project_root, "features_downloads", filename)
            features.to_csv(file_path, index=False)
            
            output_placeholder.success(f"Datos guardados en: {filename}")
            return features
            
        except Exception as e:
            output_placeholder.error(f"Error al descargar datos: {str(e)}")
            return None

def descargar_por_hash(tx_hash, output_placeholder):
    """Descarga datos de una transacción específica"""
    with st.spinner("Descargando datos de la transacción..."):
        try:
            # Obtener datos de la transacción
            output_placeholder.info(f"Consultando transacción {tx_hash}")
            tx_data = get_transaction_data(tx_hash)
            if not tx_data:
                output_placeholder.error("No se encontró la transacción")
                return None
            
            # Generar features
            output_placeholder.info("Extrayendo features...")
            features = generate_features([tx_data])
            
            # Guardar resultados
            timestamp = time.strftime("%Y_%m_%d_%H_%M")
            filename = f"features_tx_{tx_hash[:8]}_{timestamp}.csv"
            file_path = os.path.join(project_root, "features_downloads", filename)
            features.to_csv(file_path, index=False)
            
            output_placeholder.success(f"Datos guardados en: {filename}")
            return features
            
        except Exception as e:
            output_placeholder.error(f"Error al descargar datos: {str(e)}")
            return None

def main():
    st.title("Evaluación de Modelos de Detección de Fraude")
    
    # Pestañas para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs(["Evaluación de Modelos", "Descarga de Datos", "Evaluación Conjunta", "Biblioteca de Datos"])
    
    with tab1:
        # Código existente para evaluación de modelos individuales
        df = cargar_dataset_mas_reciente()
        if df is None:
            return
        
        # Guardar tx_hash antes de procesar el DataFrame
        tx_hashes = df['tx_hash'].copy() if 'tx_hash' in df.columns else None
        
        try:
            df = asegurar_columnas(df)
        except ValueError as e:
            st.error(str(e))
            return
        
        st.subheader("Selección de Modelo")
        modelo_seleccionado = st.selectbox(            "Seleccione el modelo a evaluar:",
            ['random_forest', 'logistic_regression', 'red_neuronal', 'xgboost'],
            key='modelo_seleccionado'
        )
        
        # Limpiar resultados si se cambia de modelo
        if 'ultimo_modelo_seleccionado' not in st.session_state:
            st.session_state.ultimo_modelo_seleccionado = modelo_seleccionado
        elif st.session_state.ultimo_modelo_seleccionado != modelo_seleccionado:
            st.session_state.ultimo_modelo = None
            st.session_state.ultimo_modelo_seleccionado = modelo_seleccionado
        
        try:
            modelo = cargar_modelo(f'{modelo_seleccionado}_model.joblib')
            scaler = cargar_scaler(f'{modelo_seleccionado}_scaler.joblib') if modelo_seleccionado != 'logistic_regression' else None
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return
        
        if st.button("Evaluar Modelo"):
            with st.spinner("Realizando predicciones..."):
                y_pred, y_prob, tiempo = predecir_modelo(modelo, df, modelo_seleccionado, scaler)
                
                # Guardar resultados en session_state
                if 'ultimo_modelo' not in st.session_state:
                    st.session_state.ultimo_modelo = {}
                
                st.session_state.ultimo_modelo = {
                    'modelo': modelo_seleccionado,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'tiempo': tiempo,
                    'df': df
                }
                
                st.subheader("Resultados de la Evaluación")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tiempo de ejecución", f"{tiempo:.3f} segundos")
                with col2:
                    st.metric("Fraudes detectados", sum(y_pred))
                with col3:
                    st.metric("Porcentaje de fraudes", f"{(sum(y_pred)/len(df)*100):.2f}%")
                
                st.subheader("Distribución de Probabilidades")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Determinar el rango de visualización según el modelo
                if modelo_seleccionado in ['keras', 'red_neuronal']:
                    # Modelos que usan tanh, rango [-1, 1]
                    y_prob_plot = y_prob
                    xlim_min, xlim_max = -1.05, 1.05
                    xlabel = 'Valor de Salida (tanh)'
                    umbral = 0
                    # Ajustar número de bins según el rango de valores
                    rango = y_prob_plot.max() - y_prob_plot.min()
                    n_bins = min(30, max(15, int(len(y_prob_plot) / 8)))  # Reducir número de bins para barras más anchas
                    alpha = 0.6  # Mayor transparencia para mejor visualización de superposición
                else:
                    # Otros modelos usan probabilidades [0, 1]
                    y_prob_plot = y_prob
                    xlim_min, xlim_max = -0.05, 1.05
                    xlabel = 'Probabilidad de Fraude'
                    umbral = 0.5
                    n_bins = 40
                    alpha = 0.5
                
                # Crear histograma con bins ajustados y mayor ancho de barras
                ax.hist(y_prob_plot[y_pred == 0], bins=n_bins, alpha=alpha, label='No Fraude', color='green', 
                       width=(xlim_max-xlim_min)/n_bins*0.6)  # Reducir ancho de barras a 60% del espacio
                ax.hist(y_prob_plot[y_pred == 1], bins=n_bins, alpha=alpha, label='Fraude', color='red',
                       width=(xlim_max-xlim_min)/n_bins*0.6)  # Reducir ancho de barras a 60% del espacio
                
                ax.set_title(f'Distribución de Salidas - {modelo_seleccionado.replace("_", " ").title()}')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Frecuencia')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(xlim_min, xlim_max)
                
                # Agregar línea vertical en el umbral de decisión
                ax.axvline(x=umbral, color='black', linestyle='--', alpha=0.5, label=f'Umbral ({umbral})')
                ax.legend()
                
                # Guardar la figura en session_state
                st.session_state.ultimo_modelo['fig'] = fig
                
                st.pyplot(fig)
                
                st.subheader("Información Detallada")
                st.write(f"Rango de valores: [{y_prob_plot.min():.3f}, {y_prob_plot.max():.3f}]")
                st.write(f"Media: {y_prob_plot.mean():.3f}")
                st.write(f"Desviación estándar: {y_prob_plot.std():.3f}")
                
                # Agregar información adicional para diagnóstico
                if modelo_seleccionado in ['keras', 'red_neuronal']:
                    st.write("Distribución de valores:")
                    st.write(f"- Valores < 0: {sum(y_prob_plot < 0)} ({sum(y_prob_plot < 0)/len(y_prob_plot)*100:.1f}%)")
                    st.write(f"- Valores = 0: {sum(y_prob_plot == 0)} ({sum(y_prob_plot == 0)/len(y_prob_plot)*100:.1f}%)")
                    st.write(f"- Valores > 0: {sum(y_prob_plot > 0)} ({sum(y_prob_plot > 0)/len(y_prob_plot)*100:.1f}%)")
                    st.write(f"Número de bins en el histograma: {n_bins}")

        # Botón de guardar fuera del if anterior para mantener el estado
        if 'ultimo_modelo' in st.session_state and st.session_state.ultimo_modelo:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Guardar Resultados", key="guardar_individual"):
                    modelo_actual = st.session_state.ultimo_modelo['modelo']
                    dir_resultados = os.path.join(project_root, 'resultados', f'evaluacion_{modelo_actual}')
                    
                    os.makedirs(dir_resultados, exist_ok=True)
                    
                    # Crear DataFrame con resultados
                    df_resultados = pd.DataFrame({
                        'prediccion': st.session_state.ultimo_modelo['y_pred'],
                        'probabilidad': st.session_state.ultimo_modelo['y_prob']
                    })
                    
                    # Guardar CSV
                    csv_path = os.path.join(dir_resultados, 'predicciones.csv')
                    df_resultados.to_csv(csv_path, index=False)
                    
                    # Guardar figura
                    if 'fig' in st.session_state.ultimo_modelo:
                        png_path = os.path.join(dir_resultados, 'distribucion_probabilidades.png')
                        st.session_state.ultimo_modelo['fig'].savefig(
                            png_path, 
                            dpi=300, 
                            bbox_inches='tight'
                        )
                    
                    st.success(f"Resultados guardados en: {dir_resultados}")
            
            with col2:
                # Botón para descargar CSV
                if 'ultimo_modelo' in st.session_state:
                    df_descarga = pd.DataFrame({
                        'prediccion': st.session_state.ultimo_modelo['y_pred'],
                        'probabilidad': st.session_state.ultimo_modelo['y_prob']
                    })
                    csv = df_descarga.to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV",
                        data=csv,
                        file_name=f'predicciones_{st.session_state.ultimo_modelo["modelo"]}_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                    )
            
            with col3:
                # Botón para descargar PNG
                if 'fig' in st.session_state.ultimo_modelo:
                    # Guardar temporalmente la figura
                    temp_path = os.path.join(project_root, 'resultados', 'temp.png')
                    st.session_state.ultimo_modelo['fig'].savefig(temp_path, dpi=300, bbox_inches='tight')
                    
                    with open(temp_path, 'rb') as file:
                        st.download_button(
                            label="Descargar Gráfico",
                            data=file,
                            file_name=f'distribucion_{st.session_state.ultimo_modelo["modelo"]}_{time.strftime("%Y%m%d_%H%M%S")}.png',
                            mime='image/png'
                        )
                    
                    # Eliminar archivo temporal
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    with tab2:
        st.subheader("Descarga de Datos")
        
        # Opciones de descarga
        opcion_descarga = st.radio(
            "Seleccione el tipo de descarga:",
            ["Transacciones Recientes", "Transacción por Hash"]
        )
        
        if opcion_descarga == "Transacciones Recientes":
            minutes = st.slider("Minutos hacia atrás", 1, 60, 5)
            max_tx = st.slider("Número máximo de transacciones", 10, 1000, 100)
            
            # Crear placeholder después del botón
            if st.button("Descargar Transacciones Recientes"):
                output_placeholder = st.empty()
                df = descargar_datos_recientes(minutes, max_tx, output_placeholder)
                if df is not None:
                    st.dataframe(df)
        
        else:  # Transacción por Hash
            tx_hash = st.text_input("Ingrese el hash de la transacción:")
            
            if st.button("Descargar Datos de Transacción"):
                if tx_hash:
                    output_placeholder = st.empty()
                    df = descargar_por_hash(tx_hash, output_placeholder)
                    if df is not None:
                        st.dataframe(df)
                else:
                    st.warning("Por favor, ingrese un hash de transacción válido")

    with tab3:
        st.subheader("Evaluación Conjunta de Modelos")
        
        # Cargar dataset
        df = cargar_dataset_mas_reciente()
        if df is None:
            return
        
        # Guardar tx_hash antes de procesar el DataFrame
        tx_hashes = df['tx_hash'].copy() if 'tx_hash' in df.columns else None
        
        try:
            df = asegurar_columnas(df)
        except ValueError as e:
            st.error(str(e))
            return
        
        # Inicializar variables en session_state si no existen
        if 'resultados_evaluacion' not in st.session_state:
            st.session_state.resultados_evaluacion = None
            st.session_state.df_detalles = None
            st.session_state.df_final = None
        
        # Lista de modelos disponibles
        modelos = ['random_forest', 'logistic_regression', 'red_neuronal', 'xgboost']
        
        # Opciones de selección de modelos
        modo_evaluacion = st.radio(
            "¿Cómo desea evaluar los modelos?",
            ["Todos los modelos", "Seleccionar modelos específicos"]
        )
        
        if modo_evaluacion == "Todos los modelos":
            modelos_seleccionados = modelos
            boton_texto = "Evaluar con Todos los Modelos"
        else:
            modelos_seleccionados = st.multiselect(
                "Seleccione los modelos a evaluar:",
                modelos,
                default=['random_forest'],  # Valor por defecto
                format_func=lambda x: x.replace('_', ' ').title()  # Formato más legible
            )
            if not modelos_seleccionados:
                st.warning("Por favor, seleccione al menos un modelo")
                return
            boton_texto = f"Evaluar {len(modelos_seleccionados)} Modelo{'s' if len(modelos_seleccionados) > 1 else ''}"
        
        if st.button(boton_texto):
            with st.spinner("Realizando predicciones con los modelos seleccionados..."):
                # Diccionario para almacenar resultados
                resultados = {}
                st.session_state.df_final = df.copy()
                
                # Realizar predicciones con cada modelo seleccionado
                for nombre_modelo in modelos_seleccionados:
                    try:
                        modelo = cargar_modelo(f'{nombre_modelo}_model.joblib')
                        scaler = cargar_scaler(f'{nombre_modelo}_scaler.joblib') if nombre_modelo != 'logistic_regression' else None
                        
                        y_pred, y_prob, tiempo = predecir_modelo(modelo, df, nombre_modelo, scaler)
                        
                        resultados[nombre_modelo] = {
                            'predicciones': y_pred,
                            'probabilidades': y_prob,
                            'tiempo': tiempo
                        }
                        
                        # Agregar predicciones al DataFrame final
                        st.session_state.df_final[f'pred_{nombre_modelo}'] = y_pred
                        st.session_state.df_final[f'prob_{nombre_modelo}'] = y_prob
                        
                    except Exception as e:
                        st.error(f"Error con el modelo {nombre_modelo}: {str(e)}")
                        return
                
                # Calcular votación (fraude si la mayoría de los modelos lo detectan)
                umbral_votos = len(modelos_seleccionados) // 2 + 1
                votos = sum(st.session_state.df_final[f'pred_{modelo}'] for modelo in modelos_seleccionados)
                st.session_state.df_final['voto_mayoria'] = (votos >= umbral_votos).astype(int)
                
                # Crear DataFrame con las columnas solicitadas
                df_detalles = pd.DataFrame()
                
                # Agregar columna de transacción
                if tx_hashes is not None:
                    # Mostrar solo los primeros 8 caracteres del hash
                    df_detalles['Transacción'] = tx_hashes.str[:8] + "..."
                else:
                    df_detalles['Transacción'] = [f'TX_{i+1}' for i in range(len(st.session_state.df_final))]
                
                # Agregar columnas de predicciones de cada modelo seleccionado
                for modelo in modelos_seleccionados:
                    df_detalles[f'{modelo.replace("_", " ").title()}'] = st.session_state.df_final[f'pred_{modelo}'].map({1: '🔴', 0: '🟢'})
                
                # Agregar columna de predicción final
                df_detalles['Predicción Final'] = st.session_state.df_final['voto_mayoria'].map({1: 'Fraude', 0: 'No Fraude'})
                
                # Guardar resultados en session_state
                st.session_state.resultados_evaluacion = resultados
                st.session_state.df_detalles = df_detalles
        
        # Mostrar resultados si existen
        if st.session_state.df_detalles is not None:
            
            # Mostrar resumen
            st.subheader("Resumen de Predicciones")
            # Crear DataFrame con resumen
            resumen = pd.DataFrame({
                'Modelo': modelos_seleccionados,
                'Tiempo (s)': [st.session_state.resultados_evaluacion[m]['tiempo'] for m in modelos_seleccionados],
                'Fraudes Detectados': [sum(st.session_state.resultados_evaluacion[m]['predicciones']) for m in modelos_seleccionados],
                'Porcentaje de Fraudes': [f"{(sum(st.session_state.resultados_evaluacion[m]['predicciones'])/len(df)*100):.2f}%" for m in modelos_seleccionados]
            })
            st.dataframe(resumen)
            
            # Mostrar resultados de votación
            st.subheader("Resultados de Votación")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Transacciones", len(st.session_state.df_detalles))
            with col2:
                st.metric("Fraudes Detectados", sum(st.session_state.df_detalles['Predicción Final'] == 'Fraude'))
            with col3:
                st.metric("Porcentaje de Fraudes", f"{(sum(st.session_state.df_detalles['Predicción Final'] == 'Fraude')/len(st.session_state.df_detalles)*100):.2f}%")
            
            # Mostrar distribución de votos
            st.subheader("Distribución de Votos")
            fig, ax = plt.subplots(figsize=(10, 6))
            votos = sum(st.session_state.df_final[f'pred_{modelo}'] for modelo in modelos_seleccionados)
            max_votos = len(modelos_seleccionados)
            ax.hist(votos, bins=max_votos + 1, range=(-0.5, max_votos + 0.5), alpha=0.7)
            ax.set_xlabel('Número de Modelos que Detectan Fraude')
            ax.set_ylabel('Número de Transacciones')
            ax.set_xticks(range(max_votos + 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Mostrar tabla de predicciones
            st.subheader("Tabla de Predicciones por Transacción")
            # Mostrar leyenda de interpretación
            st.info(f"""
            **Leyenda de Interpretación:**
            - 🔴 = Fraude detectado por el modelo
            - 🟢 = Transacción legítima según el modelo
            - La **Predicción Final** se determina por votación mayoritaria
            - Se considera Fraude si {umbral_votos} o más modelos lo detectan ({umbral_votos} de {len(modelos_seleccionados)})
            """)
            st.dataframe(
                st.session_state.df_detalles.style.applymap(
                    lambda x: 'background-color: #ffcdd2' if x == 'Fraude' else 'background-color: #c8e6c9',
                    subset=['Predicción Final']
                ),
                use_container_width=True,  # Hace que la tabla use todo el ancho disponible
                column_config={
                    "Transacción": st.column_config.TextColumn(
                        "Transacción",
                        width="small",
                        help="Primeros 8 caracteres del hash de la transacción"
                    )
                }
            )
            
            # Botones de descarga y guardado
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Guardar Resultados", key="guardar_conjunto"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dir_resultados = os.path.join(project_root, 'resultados', f'evaluacion_conjunta_{timestamp}')
                    os.makedirs(dir_resultados, exist_ok=True)
                    
                    # Guardar CSV de predicciones
                    st.session_state.df_detalles.to_csv(
                        os.path.join(dir_resultados, 'predicciones.csv'),
                        index=False
                    )
                    
                    # Guardar CSV de resumen
                    resumen.to_csv(
                        os.path.join(dir_resultados, 'resumen.csv'),
                        index=False
                    )
                    
                    # Guardar gráfico de distribución
                    fig.savefig(
                        os.path.join(dir_resultados, 'distribucion_votos.png'),
                        dpi=300,
                        bbox_inches='tight'
                    )
                    
                    st.success(f"Resultados guardados en: {dir_resultados}")
            
            with col2:
                # Botones para descargar CSVs
                csv_predicciones = st.session_state.df_detalles.to_csv(index=False)
                st.download_button(
                    label="Descargar Predicciones",
                    data=csv_predicciones,
                    file_name=f'predicciones_conjunto_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
            
            with col3:
                csv_resumen = resumen.to_csv(index=False)
                st.download_button(
                    label="Descargar Resumen",
                    data=csv_resumen,
                    file_name=f'resumen_conjunto_{time.strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
            
            # Botón para descargar gráfico
            temp_path = os.path.join(project_root, 'resultados', 'temp_conjunto.png')
            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            
            with open(temp_path, 'rb') as file:
                st.download_button(
                    label="Descargar Gráfico",
                    data=file,
                    file_name=f'distribucion_votos_{time.strftime("%Y%m%d_%H%M%S")}.png',
                    mime='image/png',
                    key='descargar_grafico_conjunto'
                )
            
            # Eliminar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

    with tab4:
        st.header("Biblioteca de Datos")
        subtab1, subtab2 = st.tabs(["Descargas recientes", "Resultados de Evaluación"])
        
        with subtab1:
            st.subheader("Descargas recientes")
            features_dir = os.path.join(project_root, "features_downloads")
            archivos_features = glob.glob(os.path.join(features_dir, "features_*.csv"))
            
            if archivos_features:
                # Crear una lista de archivos con sus fechas de modificación
                archivos_info = []
                for archivo in archivos_features:
                    nombre = os.path.basename(archivo)
                    fecha_mod = os.path.getmtime(archivo)
                    fecha_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fecha_mod))
                    tamaño = os.path.getsize(archivo) / 1024  # tamaño en KB
                    archivos_info.append({
                        'seleccionado': False,  # Columna para checkboxes
                        'Nombre': nombre,
                        'Fecha de modificación': fecha_str,
                        'Tamaño (KB)': f"{tamaño:.2f}"
                    })
                
                # Crear DataFrame y agregar a session_state si no existe
                if 'df_archivos' not in st.session_state:
                    st.session_state.df_archivos = pd.DataFrame(archivos_info)
                    st.session_state.df_archivos = st.session_state.df_archivos.sort_values('Fecha de modificación', ascending=False)
                
                # Crear checkboxes para cada fila
                edited_df = st.data_editor(
                    st.session_state.df_archivos,
                    column_config={
                        "seleccionado": st.column_config.CheckboxColumn(
                            "Seleccionar",
                            help="Seleccionar para descargar",
                            default=False,
                        ),
                        "Nombre": st.column_config.TextColumn(
                            "Nombre del archivo",
                            help="Nombre del archivo de características",
                        ),
                        "Fecha de modificación": st.column_config.TextColumn(
                            "Fecha de modificación",
                            help="Fecha y hora de la última modificación",
                        ),
                        "Tamaño (KB)": st.column_config.TextColumn(
                            "Tamaño",
                            help="Tamaño del archivo en kilobytes",
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Actualizar el DataFrame en session_state
                st.session_state.df_archivos = edited_df
                
                # Obtener archivos seleccionados
                archivos_seleccionados = edited_df[edited_df['seleccionado']]['Nombre'].tolist()
                
                if archivos_seleccionados:
                    if len(archivos_seleccionados) == 1:
                        # Si solo hay un archivo seleccionado, descarga simple
                        archivo_path = os.path.join(features_dir, archivos_seleccionados[0])
                        with open(archivo_path, 'rb') as f:
                            st.download_button(
                                label="Descargar archivo",
                                data=f,
                                file_name=archivos_seleccionados[0],
                                mime='text/csv'
                            )
                    else:
                        # Si hay múltiples archivos seleccionados, crear un zip
                        import io
                        import zipfile
                        
                        # Crear un buffer en memoria para el archivo ZIP
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for nombre_archivo in archivos_seleccionados:
                                archivo_path = os.path.join(features_dir, nombre_archivo)
                                zip_file.write(archivo_path, nombre_archivo)
                        
                        # Ofrecer el ZIP para descarga
                        st.download_button(
                            label=f"Descargar {len(archivos_seleccionados)} archivos seleccionados",
                            data=zip_buffer.getvalue(),
                            file_name="features_seleccionados.zip",
                            mime="application/zip"
                        )
            else:
                st.info("No hay archivos de características descargados")
        
        with subtab2:
            st.subheader("Resultados de Evaluación")
            resultados_dir = os.path.join(project_root, "resultados")
              # Mostrar resultados de evaluación para cada modelo
            modelos = ['logistic_regression', 'random_forest', 'red_neuronal', 'xgboost']
            modelo_seleccionado = st.selectbox("Seleccionar modelo:", modelos)
            
            carpeta_evaluacion = os.path.join(resultados_dir, f"evaluacion_{modelo_seleccionado}")
            if os.path.exists(carpeta_evaluacion):
                # Buscar tanto archivos CSV como PNG
                archivos_evaluacion_csv = glob.glob(os.path.join(carpeta_evaluacion, "*.csv"))
                archivos_evaluacion_png = glob.glob(os.path.join(carpeta_evaluacion, "*.png"))
                archivos_evaluacion = archivos_evaluacion_csv + archivos_evaluacion_png
                
                if archivos_evaluacion:
                    # Crear una lista de archivos con sus fechas de modificación
                    archivos_info = []
                    for archivo in archivos_evaluacion:
                        nombre = os.path.basename(archivo)
                        fecha_mod = os.path.getmtime(archivo)
                        fecha_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fecha_mod))
                        tamaño = os.path.getsize(archivo) / 1024  # tamaño en KB
                        tipo = 'CSV' if nombre.endswith('.csv') else 'PNG'
                        archivos_info.append({
                            'seleccionado': False,  # Columna para checkboxes
                            'Nombre': nombre,
                            'Tipo': tipo,
                            'Fecha de evaluación': fecha_str,
                            'Tamaño (KB)': f"{tamaño:.2f}"
                        })
                    
                    # Crear DataFrame y agregar a session_state si no existe o si cambió el modelo
                    if ('df_evaluaciones' not in st.session_state or 
                        'modelo_actual' not in st.session_state or 
                        st.session_state.modelo_actual != modelo_seleccionado):
                        st.session_state.df_evaluaciones = pd.DataFrame(archivos_info)
                        st.session_state.df_evaluaciones = st.session_state.df_evaluaciones.sort_values(['Tipo', 'Fecha de evaluación'], ascending=[True, False])
                        st.session_state.modelo_actual = modelo_seleccionado
                    
                    # Crear checkboxes para cada fila
                    edited_df = st.data_editor(
                        st.session_state.df_evaluaciones,
                        column_config={
                            "seleccionado": st.column_config.CheckboxColumn(
                                "Seleccionar",
                                help="Seleccionar para descargar",
                                default=False,
                            ),
                            "Nombre": st.column_config.TextColumn(
                                "Nombre del archivo",
                                help="Nombre del archivo de evaluación",
                            ),
                            "Tipo": st.column_config.TextColumn(
                                "Tipo",
                                help="Tipo de archivo (CSV o PNG)",
                            ),
                            "Fecha de evaluación": st.column_config.TextColumn(
                                "Fecha de evaluación",
                                help="Fecha y hora de la evaluación",
                            ),
                            "Tamaño (KB)": st.column_config.TextColumn(
                                "Tamaño",
                                help="Tamaño del archivo en kilobytes",
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Actualizar el DataFrame en session_state
                    st.session_state.df_evaluaciones = edited_df
                    
                    # Obtener archivos seleccionados
                    archivos_seleccionados = edited_df[edited_df['seleccionado']]['Nombre'].tolist()
                    
                    if archivos_seleccionados:
                        if len(archivos_seleccionados) == 1:
                            # Si solo hay un archivo seleccionado, descarga simple
                            archivo_path = os.path.join(carpeta_evaluacion, archivos_seleccionados[0])
                            with open(archivo_path, 'rb') as f:
                                mime_type = 'text/csv' if archivo_path.endswith('.csv') else 'image/png'
                                st.download_button(
                                    label="Descargar archivo seleccionado",
                                    data=f,
                                    file_name=archivos_seleccionados[0],
                                    mime=mime_type
                                )
                        else:
                            # Si hay múltiples archivos seleccionados, crear un zip
                            import io
                            import zipfile
                            
                            # Crear un buffer en memoria para el archivo ZIP
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for nombre_archivo in archivos_seleccionados:
                                    archivo_path = os.path.join(carpeta_evaluacion, nombre_archivo)
                                    zip_file.write(archivo_path, nombre_archivo)
                            

                            # Ofrecer el ZIP para descarga
                            st.download_button(
                                label=f"Descargar {len(archivos_seleccionados)} evaluaciones seleccionadas",
                                data=zip_buffer.getvalue(),
                                file_name=f"evaluaciones_{modelo_seleccionado}.zip",
                                mime="application/zip"
                            )
                else:
                    st.info(f"No hay archivos de evaluación para el modelo {modelo_seleccionado}")
            else:
                st.info(f"No hay carpeta de evaluación para el modelo {modelo_seleccionado}")


if __name__ == "__main__":
    main()