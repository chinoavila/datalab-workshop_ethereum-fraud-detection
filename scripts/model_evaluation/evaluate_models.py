import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import time
import glob
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Obtener la ruta absoluta al directorio raíz del proyecto
project_root = os.path.abspath(os.path.dirname(__file__))
if sys.argv[0] == "streamlit_app.py":
    project_root = os.path.abspath(os.path.join(project_root, os.pardir, os.pardir))
    
# Agregar el directorio de common_functions al path
sys.path.append(os.path.join(project_root, 'common_functions'))
from eval_functions import evaluar

def cargar_modelo(nombre_modelo):
    """Carga un modelo desde el directorio models"""
    ruta_models = os.path.join(project_root, 'models')
    model = joblib.load(os.path.join(ruta_models, nombre_modelo)) if nombre_modelo != 'keras' else load_model(nombre_modelo)
    return model

def cargar_scaler(nombre_scaler):
    """Carga un scaler desde el directorio models"""
    ruta_models = os.path.join(project_root, 'models')
    return joblib.load(os.path.join(ruta_models, nombre_scaler))

def asegurar_columnas(df):
    """Asegura que el DataFrame tenga las mismas columnas que se usaron en el entrenamiento"""
    # Columnas esperadas en el mismo orden que se usaron en el entrenamiento
    # Excluimos 'FLAG' ya que es la variable objetivo y no estará en los datos nuevos
    columnas_esperadas = [
        'Avg min between sent tnx',
        'Avg min between received tnx',
        'Time Diff between first and last (Mins)',
        'Sent tnx',
        'Received Tnx',
        'Number of Created Contracts',
        'Unique Received From Addresses',
        'Unique Sent To Addresses',
        'min value received',
        'max value received',
        'avg val received',
        'min val sent',
        'max val sent',
        'avg val sent',
        'min value sent to contract',
        'avg value sent to contract',
        'total transactions (including tnx to create contract',
        'total Ether sent',
        'total ether received',
        'total ether sent contracts',
        'total ether balance',
        'Total ERC20 tnxs',
        'ERC20 total Ether sent contract',
        'ERC20 uniq sent addr',
        'ERC20 uniq rec addr',
        'ERC20 uniq sent addr.1',
        'ERC20 uniq rec contract addr',
        'ERC20 min val rec',
        'ERC20 max val rec',
        'ERC20 avg val rec',
        'ERC20 avg val sent',
        'ERC20 uniq sent token name'
    ]
    
    # Verificar que todas las columnas esperadas estén presentes
    columnas_faltantes = set(columnas_esperadas) - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan las siguientes columnas en el dataset: {columnas_faltantes}")
    
    # Reordenar las columnas para que coincidan con el orden del entrenamiento
    return df[columnas_esperadas]

def predecir_modelo(modelo, X, nombre_modelo, scaler=None):
    """Realiza predicciones con un modelo y retorna las probabilidades"""
    inicio = time.time()
    
    # Aplicar scaler si existe
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Realizar predicciones
    if nombre_modelo == 'keras':
        y_pred_proba = modelo.predict(X_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        # Para Keras, tomamos la probabilidad de la clase positiva (columna 1)
        y_pred_proba = y_pred_proba[:, 1]
    elif nombre_modelo == 'red_neuronal':
        # Para la red neuronal personalizada
        W1 = modelo['W1']
        b1 = modelo['b1']
        W2 = modelo['W2']
        b2 = modelo['b2']
        FunH = modelo['FunH']
        FunO = modelo['FunO']
        
        # Propagación hacia adelante
        netasH = W1 @ X_scaled.T + b1
        salidasH = evaluar(FunH, netasH)
        netasO = W2 @ salidasH + b2
        salidasO = evaluar(FunO, netasO)
        
        # Convertir salidas a predicciones binarias
        y_pred = np.argmax(salidasO, axis=0)
        y_pred_proba = salidasO[1, :]  # Probabilidad de la clase positiva
        
        # Imprimir información de diagnóstico
        print(f"\nDiagnóstico de predicciones para {nombre_modelo}:")
        print(f"Forma de salidasO: {salidasO.shape}")
        print(f"Valores únicos en y_pred: {np.unique(y_pred, return_counts=True)}")
        print(f"Rango de probabilidades: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        
    elif nombre_modelo == 'xgboost':
        # Para XGBoost
        y_pred_proba = modelo.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Imprimir información de diagnóstico
        print(f"\nDiagnóstico de predicciones para {nombre_modelo}:")
        print(f"Valores únicos en y_pred: {np.unique(y_pred, return_counts=True)}")
        print(f"Rango de probabilidades: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        
    else:
        y_pred = modelo.predict(X_scaled)
        y_pred_proba = modelo.predict_proba(X_scaled)[:, 1]
    
    tiempo_ejecucion = time.time() - inicio
    
    return y_pred, y_pred_proba, tiempo_ejecucion

def main():
    # Buscar y cargar el dataset más reciente
    print("Buscando dataset más reciente...")
    ruta_features = os.path.join(project_root, 'features_downloads')
    archivos = glob.glob(os.path.join(ruta_features, 'features_recent_*_*.csv'))
    if not archivos:
        raise FileNotFoundError("No se encontró ningún archivo que coincida con el patrón 'features_recent_*_*.csv' en el directorio features_downloads")
    
    # Ordenar por fecha de modificación y tomar el más reciente
    archivo_mas_reciente = max(archivos, key=os.path.getmtime)
    print(f"Cargando dataset: {archivo_mas_reciente}")
    df = pd.read_csv(archivo_mas_reciente)
    
    # Asegurar que las columnas coincidan con las del entrenamiento
    print("Verificando columnas...")
    df = asegurar_columnas(df)
    
    # Cargar modelos y scalers
    print("Cargando modelos...")
    modelos = {
        'random_forest': {
            'modelo': cargar_modelo('random_forest_model.joblib'),
            'scaler': cargar_scaler('random_forest_scaler.joblib')
        },
        'keras': {
            'modelo': cargar_modelo('keras_model.joblib'),
            'scaler': cargar_scaler('keras_scaler.joblib')
        },
        'logistic_regression': {
            'modelo': cargar_modelo('logistic_regression_model.joblib'),
            'scaler': None
        },
        'red_neuronal': {
            'modelo': cargar_modelo('red_neuronal_model.joblib'),
            'scaler': cargar_scaler('red_neuronal_scaler.joblib')
        },
        'xgboost': {
            'modelo': cargar_modelo('xgboost_model.joblib'),
            'scaler': cargar_scaler('xgboost_scaler.joblib')
        }
    }
    
    # Realizar predicciones
    print("Realizando predicciones...")
    resultados = {}
    df_final = df.copy()  # Crear una copia del DataFrame original para guardar todas las predicciones
    
    for nombre_modelo, config in modelos.items():
        print(f"\nPrediciendo con {nombre_modelo}...")
        y_pred, y_pred_proba, tiempo = predecir_modelo(
            config['modelo'], 
            df,  # Usar el DataFrame original para cada modelo
            nombre_modelo, 
            config['scaler']
        )
        
        # Guardar resultados
        resultados[nombre_modelo] = {
            'predicciones': y_pred,
            'probabilidades': y_pred_proba,
            'tiempo_ejecucion': tiempo
        }
        
        # Agregar predicciones al DataFrame final
        df_final[f'pred_{nombre_modelo}'] = y_pred
        df_final[f'prob_{nombre_modelo}'] = y_pred_proba
      # Guardar resultados
    nombre_base = os.path.splitext(os.path.basename(archivo_mas_reciente))[0]
    # Crear la carpeta 'resultados' y el subdirectorio para este dataset en la raíz del proyecto
    dir_resultados = os.path.join(project_root, 'resultados', nombre_base)
    os.makedirs(dir_resultados, exist_ok=True)
    
    # Guardar DataFrame con predicciones
    df_final.to_csv(os.path.join(dir_resultados, 'predicciones.csv'), index=False)
    
    # Crear resumen de predicciones
    resumen = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'Tiempo de ejecución (s)': [r['tiempo_ejecucion'] for r in resultados.values()],
        'Fraudes detectados': [sum(r['predicciones']) for r in resultados.values()],
        'Porcentaje de fraudes': [sum(r['predicciones'])/len(df)*100 for r in resultados.values()]
    })
    
    print("\nResumen de predicciones:")
    print(resumen)
    
    # Guardar resumen
    resumen.to_csv(os.path.join(dir_resultados, 'resumen_predicciones.csv'), index=False)
    print(f"\nResultados guardados en el directorio: {dir_resultados}")
    print(f"- predicciones.csv")
    print(f"- resumen_predicciones.csv")

    # Crear visualización de predicciones
    n_modelos = len(modelos)
    n_filas = 2
    n_columnas = 3
    
    plt.figure(figsize=(15, 10))
    
    # Crear archivo de diagnóstico
    with open(os.path.join(dir_resultados, 'diagnostico.txt'), 'w') as f:
        f.write(f"Diagnóstico de evaluación para dataset: {nombre_base}\n")
        f.write("="*80 + "\n\n")
        
        # Crear subplot para cada modelo
        for i, (nombre_modelo, config) in enumerate(modelos.items()):
            plt.subplot(n_filas, n_columnas, i+1)
            
            # Obtener predicciones y probabilidades
            y_pred = df_final[f'pred_{nombre_modelo}']
            y_prob = df_final[f'prob_{nombre_modelo}']
            
            # Determinar el rango de visualización según el modelo
            if nombre_modelo == 'keras':
                # Keras usa tanh, rango [-1, 1]
                y_prob_plot = y_prob
                xlim_min, xlim_max = -1.05, 1.05
                xlabel = 'Valor de Salida (tanh)'
            elif nombre_modelo == 'red_neuronal':
                # Red neuronal personalizada usa tanh, rango [-1, 1]
                y_prob_plot = y_prob
                xlim_min, xlim_max = -1.05, 1.05
                xlabel = 'Valor de Salida (tanh)'
            else:
                # Otros modelos usan probabilidades [0, 1]
                y_prob_plot = y_prob
                xlim_min, xlim_max = -0.05, 1.05
                xlabel = 'Probabilidad de Fraude'
            
            # Escribir información de diagnóstico
            f.write(f"\nDiagnóstico para {nombre_modelo}:\n")
            f.write("-"*50 + "\n")
            f.write(f"Número total de casos: {len(y_pred)}\n")
            f.write(f"Casos de no fraude: {sum(y_pred == 0)}\n")
            f.write(f"Casos de fraude: {sum(y_pred == 1)}\n")
            f.write(f"Porcentaje de fraudes: {(sum(y_pred == 1)/len(y_pred)*100):.2f}%\n")
            f.write(f"Rango de probabilidades: [{y_prob_plot.min():.3f}, {y_prob_plot.max():.3f}]\n")
            f.write(f"Media de probabilidades: {y_prob_plot.mean():.3f}\n")
            f.write(f"Desviación estándar de probabilidades: {y_prob_plot.std():.3f}\n")
            
            # Crear histograma de probabilidades
            plt.hist(y_prob_plot[y_pred == 0], bins=40, alpha=0.5, label='No Fraude', color='green')
            plt.hist(y_prob_plot[y_pred == 1], bins=40, alpha=0.5, label='Fraude', color='red')
            
            plt.title(f'Distribución de Salidas\n{nombre_modelo.replace("_", " ").title()}')
            plt.xlabel(xlabel)
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, alpha=0.9)
            
            # Ajustar límites del eje y para mejor visualización
            plt.ylim(0, max(plt.ylim()[1], 1))
            
            # Ajustar límites del eje x según el modelo
            plt.xlim(xlim_min, xlim_max)
            
            # Agregar línea vertical en el umbral de decisión
            if nombre_modelo in ['keras', 'red_neuronal']:
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Umbral (0)')
            else:
                plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Umbral (0.5)')
            plt.legend()
        
        # Escribir resumen general
        f.write("\n\nResumen General:\n")
        f.write("="*50 + "\n")
        for nombre_modelo in modelos.keys():
            y_pred = df_final[f'pred_{nombre_modelo}']
            f.write(f"\n{nombre_modelo.replace('_', ' ').title()}:\n")
            f.write(f"- Tiempo de ejecución: {resultados[nombre_modelo]['tiempo_ejecucion']:.2f} segundos\n")
            f.write(f"- Fraudes detectados: {sum(y_pred == 1)}\n")
            f.write(f"- Porcentaje de fraudes: {(sum(y_pred == 1)/len(y_pred)*100):.2f}%\n")
    
    # Ajustar el espacio entre subplots
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig(os.path.join(dir_resultados, 'distribucion_probabilidades.png'), dpi=300, bbox_inches='tight')
    print(f"\nResultados guardados en el directorio: {dir_resultados}")
    print(f"- distribucion_probabilidades.png")
    print(f"- diagnostico.txt")
    
    # Mostrar el gráfico
    plt.show()

if __name__ == "__main__":
    main() 