#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para predecir propiedades geotécnicas a mayores profundidades.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_processed_data():
    """
    Carga los datos procesados.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con los datos procesados
    """
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_path = project_dir / "data" / "processed" / "datos_procesados.csv"
    
    if os.path.exists(processed_data_path):
        return pd.read_csv(processed_data_path)
    else:
        print(f"No se encontró el archivo {processed_data_path}")
        return None

def prepare_data_for_prediction(df, target_property='Vs (m/s)'):
    """
    Prepara los datos para el modelo predictivo.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
    target_property : str, optional
        Propiedad objetivo a predecir
        
    Returns:
    --------
    tuple
        (X, y) características y objetivo
    """
    # Asegurarse que la columna objetivo existe
    if target_property not in df.columns:
        print(f"La propiedad {target_property} no existe en los datos")
        return None, None
    
    # Usar profundidad como característica principal
    if 'profundidad_media' in df.columns:
        X = df[['profundidad_media']].copy()
    elif 'De' in df.columns and 'Hasta' in df.columns:
        df['profundidad_media'] = (df['De'] + df['Hasta']) / 2
        X = df[['profundidad_media']].copy()
    else:
        print("No hay información de profundidad disponible")
        return None, None
    
    # Agregar características adicionales si están disponibles
    additional_features = ['Gravas', 'Arenas', 'Finos', 'W%']
    for feature in additional_features:
        if feature in df.columns:
            # Imputar valores faltantes con la media
            mean_value = df[feature].mean()
            X[feature] = df[feature].fillna(mean_value)
    
    # Objetivo
    y = df[target_property].dropna()
    
    # Filtrar X para coincidir con y
    X = X.loc[y.index]
    
    return X, y

def train_depth_model(X, y):
    """
    Entrena un modelo para predecir propiedades en función de la profundidad.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Características
    y : pandas.Series
        Objetivo
        
    Returns:
    --------
    dict
        Resultados del modelo
    """
    # Verificar datos suficientes
    if len(X) < 10:
        print("Datos insuficientes para entrenar el modelo")
        return None
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        max_depth=5
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Resultados
    results = {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'rmse': rmse,
        'feature_importance': feature_importance
    }
    
    return results

def predict_property_at_depth(results, max_depth=100, step=0.5):
    """
    Predice la propiedad a diferentes profundidades.
    
    Parameters:
    -----------
    results : dict
        Resultados del modelo
    max_depth : float, optional
        Profundidad máxima para la predicción
    step : float, optional
        Paso para las profundidades
        
    Returns:
    --------
    pandas.DataFrame
        Predicciones a diferentes profundidades
    """
    model = results['model']
    scaler = results['scaler']
    
    # Generar rango de profundidades
    depths = np.arange(0, max_depth + step, step)
    
    # Crear DataFrame para las predicciones
    X_pred = pd.DataFrame({'profundidad_media': depths})
    
    # Agregar valores medios para otras características si se usaron
    n_features = len(scaler.mean_)
    
    if n_features > 1:
        # Obtener nombres de características del modelo
        feature_names = results['feature_importance']['feature'].tolist()
        
        # Añadir valores medios para características adicionales
        for i in range(1, n_features):
            feature_name = feature_names[i]
            X_pred[feature_name] = scaler.mean_[i]
    
    # Escalar datos
    X_pred_scaled = scaler.transform(X_pred)
    
    # Realizar predicción
    predictions = model.predict(X_pred_scaled)
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'profundidad': depths,
        'prediccion': predictions
    })
    
    return results_df

def visualize_depth_prediction(df, predictions, property_name):
    """
    Visualiza la predicción de la propiedad con la profundidad.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos originales
    predictions : pandas.DataFrame
        Predicciones a diferentes profundidades
    property_name : str
        Nombre de la propiedad
    """
    # Determinar la columna de profundidad en los datos originales
    if 'profundidad_media' in df.columns:
        depth_col = 'profundidad_media'
    else:
        depth_col = 'De'  # Usar 'De' como aproximación
    
    # Obtener datos reales
    real_data = df[[depth_col, property_name]].dropna()
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar datos reales
    plt.scatter(
        real_data[depth_col],
        real_data[property_name],
        c='blue',
        label='Datos medidos',
        alpha=0.7,
        s=60
    )
    
    # Graficar predicciones
    plt.plot(
        predictions['profundidad'],
        predictions['prediccion'],
        'r-',
        label='Predicción',
        linewidth=2
    )
    
    # Sombrear área de extrapolación
    max_measured_depth = real_data[depth_col].max()
    plt.axvspan(
        max_measured_depth, 
        predictions['profundidad'].max(),
        alpha=0.1,
        color='gray',
        label=f'Extrapolación (>{max_measured_depth}m)'
    )
    
    # Añadir etiquetas y título
    plt.xlabel('Profundidad (m)', fontsize=12)
    plt.ylabel(f'{property_name}', fontsize=12)
    plt.title(f'Predicción de {property_name} vs Profundidad', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ajustar límites
    plt.xlim(0, predictions['profundidad'].max() * 1.05)
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "profiles"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    property_filename = property_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    plt.savefig(results_dir / f"prediccion_{property_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figura guardada en {results_dir / f'prediccion_{property_filename}.png'}")

def main():
    """Función principal"""
    # Cargar datos procesados
    df = load_processed_data()
    
    if df is not None:
        print("Preparando modelo predictivo...")
        
        # 1. Modelo para Velocidad S (Vs)
        property_name = 'Vs (m/s)'
        if property_name in df.columns:
            print(f"Entrenando modelo para {property_name}...")
            
            # Preparar datos
            X, y = prepare_data_for_prediction(df, property_name)
            
            if X is not None and y is not None:
                # Entrenar modelo
                results = train_depth_model(X, y)
                
                if results is not None:
                    # Realizar predicciones
                    predictions = predict_property_at_depth(results, max_depth=100)
                    
                    # Visualizar predicciones
                    visualize_depth_prediction(df, predictions, property_name)
                    
                    # Guardar modelo
                    import joblib
                    
                    models_dir = Path(__file__).resolve().parents[2] / "models" / "trained"
                    models_dir.mkdir(exist_ok=True, parents=True)
                    
                    joblib.dump(results, models_dir / f"modelo_{property_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl")
                    
                    # Guardar predicciones
                    results_dir = Path(__file__).resolve().parents[2] / "results" / "predictions"
                    results_dir.mkdir(exist_ok=True, parents=True)
                    
                    predictions.to_csv(results_dir / f"predicciones_{property_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv", index=False)
                    
                    print(f"Análisis predictivo para {property_name} completado con éxito!")
                else:
                    print(f"No se pudo entrenar el modelo para {property_name}.")
            else:
                print(f"No hay suficientes datos para entrenar el modelo de {property_name}.")
        else:
            print(f"La propiedad {property_name} no existe en los datos.")
            
        # 2. Si hay datos de NSPT, entrenar modelo para NSPT
        property_name = 'Nspt'
        if property_name in df.columns:
            print(f"Entrenando modelo para {property_name}...")
            
            # Filtrar datos SPT
            df_spt = df[df['tipo_ensayo'] == 'SPT'].copy()
            
            if not df_spt.empty:
                # Preparar datos
                X, y = prepare_data_for_prediction(df_spt, property_name)
                
                if X is not None and y is not None and len(X) >= 10:
                    # Entrenar modelo
                    results = train_depth_model(X, y)
                    
                    if results is not None:
                        # Realizar predicciones
                        predictions = predict_property_at_depth(results, max_depth=100)
                        
                        # Visualizar predicciones
                        visualize_depth_prediction(df_spt, predictions, property_name)
                        
                        # Guardar modelo y predicciones (similar al caso de Vs)
                        print(f"Análisis predictivo para {property_name} completado con éxito!")
            else:
                print(f"No hay suficientes datos de SPT para entrenar el modelo de {property_name}.")
                
        print("Análisis predictivo completado!")
    else:
        print("No se pudo cargar los datos procesados.")

if __name__ == "__main__":
    main()