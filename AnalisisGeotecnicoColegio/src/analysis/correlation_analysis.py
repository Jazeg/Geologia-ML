#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analizar correlaciones entre propiedades geotécnicas,
especialmente Velocidad S (Vs) y NSPT.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.gridspec as gridspec

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

def create_correlation_data(df_mw, df_spt):
    """
    Crea un DataFrame para correlacionar Vs con NSPT.
    
    Parameters:
    -----------
    df_mw : pandas.DataFrame
        DataFrame con datos de ensayos MW (Vs)
    df_spt : pandas.DataFrame
        DataFrame con datos de ensayos SPT (NSPT)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con datos para correlación
    """
    # Este proceso requiere interpolación y correlación espacial
    # Para simplificar, crearemos un dataset sintético basado en las profundidades
    # En un caso real, se necesitaría un método más sofisticado de interpolación espacial
    
    correlation_data = []
    
    # Obtener rangos de profundidad
    min_depth = min(df_mw['De'].min(), df_spt['De'].min())
    max_depth = max(df_mw['Hasta'].max(), df_spt['Hasta'].max())
    
    # Crear intervalos de profundidad
    depth_intervals = np.linspace(min_depth, max_depth, 20)
    
    for depth in depth_intervals:
        # Filtrar datos cercanos a esta profundidad
        mw_near = df_mw[(df_mw['De'] <= depth) & (df_mw['Hasta'] >= depth)]
        spt_near = df_spt[(df_spt['De'] <= depth) & (df_spt['Hasta'] >= depth)]
        
        if not mw_near.empty and not spt_near.empty:
            # Calcular valores promedio para esta profundidad
            avg_vs = mw_near['Vs (m/s)'].mean()
            avg_nspt = spt_near['Nspt'].mean()
            
            if not pd.isna(avg_vs) and not pd.isna(avg_nspt):
                correlation_data.append({
                    'profundidad': depth,
                    'Vs': avg_vs,
                    'NSPT': avg_nspt
                })
    
    return pd.DataFrame(correlation_data)

def analyze_vs_nspt_correlation(corr_data):
    """
    Analiza la correlación entre Vs y NSPT.
    
    Parameters:
    -----------
    corr_data : pandas.DataFrame
        DataFrame con datos para correlación
        
    Returns:
    --------
    dict
        Diccionario con resultados del análisis
    """
    # Verificar datos suficientes
    if len(corr_data) < 5:
        print("Datos insuficientes para análisis de correlación")
        return None
    
    # Calcular correlación
    correlation = corr_data[['Vs', 'NSPT']].corr().iloc[0, 1]
    
    # Ajustar modelo lineal
    X = corr_data[['NSPT']]
    y = corr_data['Vs']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    
    # Métricas
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Coeficientes
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Ecuación
    equation = f"Vs = {slope:.2f} × NSPT + {intercept:.2f}"
    
    # Resultados
    results = {
        'correlation': correlation,
        'r2': r2,
        'rmse': rmse,
        'equation': equation,
        'model': model
    }
    
    return results

def visualize_vs_nspt_correlation(corr_data, results):
    """
    Visualiza la correlación entre Vs y NSPT.
    
    Parameters:
    -----------
    corr_data : pandas.DataFrame
        DataFrame con datos para correlación
    results : dict
        Diccionario con resultados del análisis
    """
    # Crear figura
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Gráfico principal: Vs vs NSPT
    ax1 = plt.subplot(gs[0, :])
    sns.regplot(
        x='NSPT', 
        y='Vs', 
        data=corr_data,
        scatter_kws={'s': 80, 'alpha': 0.7},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax1
    )
    
    # Añadir texto con resultados
    equation = results['equation']
    r2 = results['r2']
    text = f"{equation}\nR² = {r2:.3f}"
    
    ax1.text(
        0.05, 0.95, 
        text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax1.set_title('Correlación entre Velocidad S y NSPT', fontsize=16)
    ax1.set_xlabel('NSPT (número de golpes)', fontsize=12)
    ax1.set_ylabel('Velocidad S (m/s)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico inferior izquierdo: Vs vs Profundidad
    ax2 = plt.subplot(gs[1, 0])
    sns.scatterplot(
        x='profundidad',
        y='Vs',
        data=corr_data,
        s=60,
        ax=ax2
    )
    ax2.set_title('Velocidad S vs Profundidad', fontsize=12)
    ax2.set_xlabel('Profundidad (m)', fontsize=10)
    ax2.set_ylabel('Velocidad S (m/s)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico inferior derecho: NSPT vs Profundidad
    ax3 = plt.subplot(gs[1, 1])
    sns.scatterplot(
        x='profundidad',
        y='NSPT',
        data=corr_data,
        s=60,
        ax=ax3
    )
    ax3.set_title('NSPT vs Profundidad', fontsize=12)
    ax3.set_xlabel('Profundidad (m)', fontsize=10)
    ax3.set_ylabel('NSPT (número de golpes)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "correlations"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / "correlacion_vs_nspt.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figura guardada en {results_dir / 'correlacion_vs_nspt.png'}")

def analyze_properties_correlation(df):
    """
    Analiza correlaciones entre todas las propiedades geotécnicas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
    """
    # Seleccionar propiedades numéricas relevantes
    numeric_cols = [
        'profundidad_media', 'potencia', 'Vs (m/s)', 'Vp (m/s)', 
        'Nspt', '(N1)60', 'LL %', 'LP', 'IP %', 'W%',
        'Gravas', 'Arenas', 'Finos'
    ]
    
    # Filtrar columnas que existen en el DataFrame
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    # Eliminar filas con todos los valores NaN en estas columnas
    df_numeric = df[valid_cols].dropna(how='all')
    
    # Calcular matriz de correlación
    corr_matrix = df_numeric.corr()
    
    # Crear figura
    plt.figure(figsize=(14, 12))
    
    # Crear máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Crear mapa de calor
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Matriz de Correlación de Propiedades Geotécnicas', fontsize=16)
    plt.tight_layout()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "correlations"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / "matriz_correlacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz de correlación guardada en {results_dir / 'matriz_correlacion.png'}")
    
    return corr_matrix

def main():
    """Función principal"""
    # Cargar datos procesados
    df = load_processed_data()
    
    if df is not None:
        print("Analizando correlaciones...")
        
        # Dividir por tipo de ensayo
        df_mw = df[df['tipo_ensayo'] == 'MW']
        df_spt = df[df['tipo_ensayo'] == 'SPT']
        
        # Análisis general de correlaciones
        corr_matrix = analyze_properties_correlation(df)
        
        # Verificar si hay suficientes datos para correlacionar Vs y NSPT
        if not df_mw.empty and not df_spt.empty:
            print("Analizando correlación Vs-NSPT...")
            
            # Crear datos de correlación
            corr_data = create_correlation_data(df_mw, df_spt)
            
            if len(corr_data) > 0:
                # Analizar correlación
                results = analyze_vs_nspt_correlation(corr_data)
                
                if results is not None:
                    # Visualizar correlación
                    visualize_vs_nspt_correlation(corr_data, results)
                    
                    # Guardar resultados
                    results_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
                    results_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Guardar datos de correlación
                    corr_data.to_csv(results_dir / "datos_correlacion_vs_nspt.csv", index=False)
                    
                    # Guardar resultados en un archivo de texto
                    with open(results_dir / "resultados_correlacion_vs_nspt.txt", 'w') as f:
                        f.write(f"Correlación Vs-NSPT: {results['correlation']:.3f}\n")
                        f.write(f"R²: {results['r2']:.3f}\n")
                        f.write(f"RMSE: {results['rmse']:.2f} m/s\n")
                        f.write(f"Ecuación: {results['equation']}\n")
                    
                    print("Análisis de correlación Vs-NSPT completado con éxito!")
                else:
                    print("No se pudo completar el análisis de correlación.")
            else:
                print("No hay suficientes datos para correlacionar Vs y NSPT.")
        else:
            print("No hay suficientes datos de ensayos MW o SPT para análisis de correlación.")
            
        print("Análisis de correlaciones completado!")
    else:
        print("No se pudo cargar los datos procesados.")

if __name__ == "__main__":
    main()