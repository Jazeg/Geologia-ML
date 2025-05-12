#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para cargar y procesar los datos geotécnicos del Colegio María Ignacia.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_geotechnical_data(filepath, delimiter=';'):
    """
    Carga los datos geotécnicos desde un archivo CSV.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV
    delimiter : str, optional
        Delimitador utilizado en el CSV (por defecto ';')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con los datos cargados
    """
    try:
        # Probar diferentes codificaciones
        encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                
                # Eliminar filas vacías al final del archivo
                df = df.dropna(how='all')
                
                # Verificar si se cargó correctamente
                if len(df.columns) > 1 and len(df) > 0:
                    print(f"Archivo cargado con éxito usando codificación: {encoding}")
                    print(f"Dimensiones: {df.shape}")
                    return df
            except UnicodeDecodeError:
                continue
                
        raise Exception("No se pudo decodificar el archivo con las codificaciones disponibles")
    
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def process_data(df):
    """
    Procesa y limpia los datos geotécnicos.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos cargados
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame procesado
    """
    # Hacer una copia para no modificar el original
    processed_df = df.copy()
    
    # Reemplazar '-' por NaN
    processed_df = processed_df.replace('-', np.nan)
    
    # Convertir columnas numéricas
    numeric_cols = ['Norte', 'Este', 'De', 'Hasta', 'potencia', 
                    'Vs (m/s)', 'Vp (m/s)', 'Nspt', '(N1)60', 
                    'LL %', 'LP', 'IP %', 'W%', 'C (Kg/cm2)', 
                    '? h�meda (gr/cm3)', '? seca (gr/cm3)', 
                    'Qadm. (kg/cm2)', 'Qult. (kg/cm2)',
                    'SST (ppm)', 'SO4 (ppm)', 'CL (ppm)', 'pH (ppm)',
                    'Gravas', 'Arenas', 'Finos']
    
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Identificar tipo de ensayo
    processed_df['tipo_ensayo'] = processed_df['�tem'].str.split('-').str[0]
    
    # Crear columna de profundidad media
    if 'De' in processed_df.columns and 'Hasta' in processed_df.columns:
        processed_df['profundidad_media'] = (processed_df['De'] + processed_df['Hasta']) / 2
    
    return processed_df

def split_by_test_type(df):
    """
    Divide el DataFrame por tipo de ensayo.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
        
    Returns:
    --------
    dict
        Diccionario con DataFrames separados por tipo de ensayo
    """
    # Identificar los tipos de ensayo disponibles
    test_types = df['tipo_ensayo'].dropna().unique()
    
    # Crear un diccionario para almacenar los DataFrames
    dfs_by_type = {}
    
    for test_type in test_types:
        dfs_by_type[test_type] = df[df['tipo_ensayo'] == test_type].copy()
    
    return dfs_by_type

def visualize_test_locations(df):
    """
    Visualiza la ubicación espacial de los ensayos.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
    """
    if 'Norte' not in df.columns or 'Este' not in df.columns:
        print("No se pueden visualizar las ubicaciones: faltan coordenadas")
        return
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Colores por tipo de ensayo
    colors = {'SPT': 'red', 'C': 'blue', 'MW': 'green'}
    
    # Graficar puntos
    for test_type, group in df.groupby('tipo_ensayo'):
        plt.scatter(
            group['Este'], 
            group['Norte'], 
            c=colors.get(test_type, 'gray'),
            label=test_type,
            s=100,
            alpha=0.7
        )
    
    # Añadir etiquetas a los puntos
    for _, row in df.drop_duplicates('�tem').iterrows():
        plt.annotate(
            row['�tem'],
            (row['Este'], row['Norte']),
            fontsize=8,
            ha='right',
            va='bottom'
        )
    
    plt.title('Ubicación de Ensayos - Colegio María Ignacia')
    plt.xlabel('Este')
    plt.ylabel('Norte')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "maps"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / "ubicacion_ensayos.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figura guardada en {results_dir / 'ubicacion_ensayos.png'}")

def main():
    """Función principal"""
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Definir rutas
    raw_data_path = project_dir / "data" / "raw" / "Colegio1MariaIgnacia.csv"
    processed_data_path = project_dir / "data" / "processed" / "datos_procesados.csv"
    interim_data_dir = project_dir / "data" / "interim"
    
    # Crear directorios si no existen
    interim_data_dir.mkdir(exist_ok=True, parents=True)
    processed_data_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Cargar datos
    print(f"Cargando datos desde: {raw_data_path}")
    df = load_geotechnical_data(raw_data_path)
    
    if df is not None:
        # Procesar datos
        print("Procesando datos...")
        processed_df = process_data(df)
        
        # Guardar datos procesados
        processed_df.to_csv(processed_data_path, index=False, encoding='utf-8')
        print(f"Datos procesados guardados en: {processed_data_path}")
        
        # Dividir por tipo de ensayo
        print("Dividiendo datos por tipo de ensayo...")
        dfs_by_type = split_by_test_type(processed_df)
        
        # Guardar cada tipo de ensayo por separado
        for test_type, test_df in dfs_by_type.items():
            output_path = interim_data_dir / f"ensayo_{test_type}.csv"
            test_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Datos de ensayo {test_type} guardados en: {output_path}")
        
        # Visualizar ubicaciones
        print("Generando mapa de ubicaciones...")
        visualize_test_locations(processed_df)
        
        print("Procesamiento completado con éxito!")
    else:
        print("No se pudo procesar los datos. Verifique el archivo de entrada.")

if __name__ == "__main__":
    main()