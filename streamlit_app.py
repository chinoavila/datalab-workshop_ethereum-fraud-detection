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
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    
    # Crear DataFrame con métricas
    df_metricas = pd.DataFrame({
        'Métrica': list(metricas.keys()),
        'Valor': [f"{v:.3f}" for v in metricas.values()]
    })
    
    return df_metricas

def descargar_datos_recientes(minutes, max_tx):
    """Descarga datos de transacciones recientes y su histórico"""
    with st.spinner("Descargando transacciones recientes..."):
        try:
            # Obtener transacciones recientes
            recent_tx = get_recent_transactions(minutes=minutes, max_tx=max_tx)
            if not recent_tx:
                st.error("No se encontraron transacciones recientes")
                return None
            
            st.info(f"Se obtuvieron {len(recent_tx)} transacciones recientes")
            
            # Obtener direcciones únicas de remitentes
            addresses = set()
            for tx in recent_tx:
                if tx.get("from"):
                    addresses.add(tx["from"])
            
            st.info(f"Consultando histórico para {len(addresses)} direcciones...")
            
            # Obtener histórico para cada dirección
            all_hist_txs = recent_tx.copy()
            for i, addr in enumerate(addresses):
                st.text(f"Consultando histórico para {addr} ({i+1}/{len(addresses)})")
                hist_txs = get_historical_transfers_for_address(addr, max_tx=max_tx)
                all_hist_txs.extend(hist_txs)
                time.sleep(0.25)  # Evitar rate limiting
            
            st.info(f"Total de transacciones históricas recopiladas: {len(all_hist_txs)}")
            
            # Generar features
            st.text("Extrayendo features...")
            features = generate_features(all_hist_txs, recent_tx)
              # Guardar resultados
            timestamp = time.strftime("%Y_%m_%d_%H_%M")
            filename = f"features_recent_{minutes}m_{max_tx}tx_{timestamp}.csv"
            file_path = os.path.join(project_root, "features_downloads", filename)
            features.to_csv(file_path, index=False)
            
            st.success(f"Datos guardados en: {filename}")
            return features
            
        except Exception as e:
            st.error(f"Error al descargar datos: {str(e)}")
            return None

def descargar_por_hash(tx_hash):
    """Descarga datos de una transacción específica"""
    with st.spinner("Descargando datos de la transacción..."):
        try:
            # Obtener datos de la transacción
            tx_data = get_transaction_data(tx_hash)
            if not tx_data:
                st.error("No se encontró la transacción")
                return None
            
            # Generar features
            features = generate_features([tx_data])
              # Guardar resultados
            timestamp = time.strftime("%Y_%m_%d_%H_%M")
            filename = f"features_tx_{tx_hash[:8]}_{timestamp}.csv"
            file_path = os.path.join(project_root, "features_downloads", filename)
            features.to_csv(file_path, index=False)
            
            st.success(f"Datos guardados en: {filename}")
            return features
            
        except Exception as e:
            st.error(f"Error al descargar datos: {str(e)}")
            return None

def main():
    st.title("Evaluación de Modelos de Detección de Fraude")
    
    # Pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Evaluación de Modelos", "Descarga de Datos", "Evaluación Conjunta"])
    
    with tab1:
        # Código existente para evaluación de modelos individuales
        df = cargar_dataset_mas_reciente()
        if df is None:
            return
        
        try:
            df = asegurar_columnas(df)
        except ValueError as e:
            st.error(str(e))
            return
        
        st.subheader("Selección de Modelo")
        modelo_seleccionado = st.selectbox(
            "Seleccione el modelo a evaluar:",
            ['random_forest', 'keras', 'logistic_regression', 'red_neuronal', 'xgboost']
        )
        
        try:
            modelo = cargar_modelo(f'{modelo_seleccionado}_model.joblib')
            scaler = cargar_scaler(f'{modelo_seleccionado}_scaler.joblib') if modelo_seleccionado != 'logistic_regression' else None
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return
        
        if st.button("Evaluar Modelo"):
            with st.spinner("Realizando predicciones..."):
                y_pred, y_prob, tiempo = predecir_modelo(modelo, df, modelo_seleccionado, scaler)
                
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

            if st.button("Guardar Resultados"):
                dir_resultados = os.path.join(project_root, 'resultados', f'evaluacion_{modelo_seleccionado}')

                output_placeholder.info(dir_resultados)

                os.makedirs(dir_resultados, exist_ok=True)
                
                df_resultados = pd.DataFrame({
                    'prediccion': y_pred,
                    'probabilidad': y_prob
                })
                df_resultados.to_csv(os.path.join(dir_resultados, 'predicciones.csv'), index=False)
                
                fig.savefig(os.path.join(dir_resultados, 'distribucion_probabilidades.png'), dpi=300, bbox_inches='tight')
                
                st.success(f"Resultados guardados en: {dir_resultados}")
    
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
            
            if st.button("Descargar Transacciones Recientes"):
                df = descargar_datos_recientes(minutes, max_tx)
                if df is not None:
                    st.dataframe(df)
        
        else:  # Transacción por Hash
            tx_hash = st.text_input("Ingrese el hash de la transacción:")
            
            if st.button("Descargar Datos de Transacción"):
                if tx_hash:
                    df = descargar_por_hash(tx_hash)
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
        
        try:
            df = asegurar_columnas(df)
        except ValueError as e:
            st.error(str(e))
            return
        
        if st.button("Evaluar con Todos los Modelos"):
            with st.spinner("Realizando predicciones con todos los modelos..."):
                # Diccionario para almacenar resultados
                resultados = {}
                df_final = df.copy()
                
                # Lista de modelos a evaluar
                modelos = ['random_forest', 'keras', 'logistic_regression', 'red_neuronal', 'xgboost']
                
                # Realizar predicciones con cada modelo
                for nombre_modelo in modelos:
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
                        df_final[f'pred_{nombre_modelo}'] = y_pred
                        df_final[f'prob_{nombre_modelo}'] = y_prob
                        
                    except Exception as e:
                        st.error(f"Error con el modelo {nombre_modelo}: {str(e)}")
                        return
                
                # Calcular votación (fraude si al menos 3 modelos lo detectan)
                votos = sum(df_final[f'pred_{modelo}'] for modelo in modelos)
                df_final['voto_mayoria'] = (votos >= 3).astype(int)
                
                # Mostrar resumen
                st.subheader("Resumen de Predicciones")
                
                # Crear DataFrame con resumen
                resumen = pd.DataFrame({
                    'Modelo': modelos,
                    'Tiempo (s)': [resultados[m]['tiempo'] for m in modelos],
                    'Fraudes Detectados': [sum(resultados[m]['predicciones']) for m in modelos],
                    'Porcentaje de Fraudes': [f"{(sum(resultados[m]['predicciones'])/len(df)*100):.2f}%" for m in modelos]
                })
                
                st.dataframe(resumen)
                
                # Mostrar resultados de votación
                st.subheader("Resultados de Votación")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Transacciones", len(df))
                with col2:
                    st.metric("Fraudes por Votación", sum(df_final['voto_mayoria']))
                with col3:
                    st.metric("Porcentaje de Fraudes", f"{(sum(df_final['voto_mayoria'])/len(df)*100):.2f}%")
                
                # Mostrar distribución de votos
                st.subheader("Distribución de Votos")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(votos, bins=6, range=(-0.5, 5.5), alpha=0.7)
                ax.set_xlabel('Número de Modelos que Detectan Fraude')
                ax.set_ylabel('Número de Transacciones')
                ax.set_xticks(range(6))
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Mostrar detalles de las transacciones
                st.subheader("Detalles de Transacciones")
                if st.checkbox("Mostrar Transacciones Detectadas como Fraude"):
                    df_fraudes = df_final[df_final['voto_mayoria'] == 1]
                    st.dataframe(df_fraudes)
                
                # Guardar resultados
                if st.button("Guardar Resultados de Votación"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dir_resultados = os.path.join(project_root, 'resultados', f'votacion_{timestamp}')
                    os.makedirs(dir_resultados, exist_ok=True)
                    
                    # Guardar DataFrame con todas las predicciones
                    df_final.to_csv(os.path.join(dir_resultados, 'predicciones_votacion.csv'), index=False)
                    
                    # Guardar resumen
                    resumen.to_csv(os.path.join(dir_resultados, 'resumen_votacion.csv'), index=False)
                    
                    # Guardar gráfico
                    fig.savefig(os.path.join(dir_resultados, 'distribucion_votos.png'), dpi=300, bbox_inches='tight')
                    
                    st.success(f"Resultados guardados en: {dir_resultados}")

if __name__ == "__main__":
    main() 