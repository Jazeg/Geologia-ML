#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para desarrollar modelos de correlación entre Velocidad S (Vs) y NSPT.
Implementa múltiples modelos de regresión para encontrar la mejor correlación.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

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

def create_correlation_data(df_mw, df_spt, max_distance=10.0):
    """
    Crea un DataFrame para correlacionar Vs con NSPT considerando
    la posición espacial de los ensayos.
    
    Parameters:
    -----------
    df_mw : pandas.DataFrame
        DataFrame con datos de ensayos MW (Vs)
    df_spt : pandas.DataFrame
        DataFrame con datos de ensayos SPT (NSPT)
    max_distance : float, optional
        Distancia máxima (en metros) para considerar ensayos cercanos
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con datos para correlación
    """
    correlation_data = []
    
    # Verificar columnas necesarias
    required_cols = ['Norte', 'Este', 'De', 'Hasta', 'Vs (m/s)', 'Nspt']
    missing_mw = [col for col in ['Norte', 'Este', 'De', 'Hasta', 'Vs (m/s)'] if col not in df_mw.columns]
    missing_spt = [col for col in ['Norte', 'Este', 'De', 'Hasta', 'Nspt'] if col not in df_spt.columns]
    
    if missing_mw or missing_spt:
        print(f"Faltan columnas para correlación espacial: MW {missing_mw}, SPT {missing_spt}")
        # Usar método alternativo basado solo en profundidad
        return create_correlation_data_by_depth(df_mw, df_spt)
    
    # Calcular profundidad media para cada muestra si no existe
    if 'profundidad_media' not in df_mw.columns:
        df_mw['profundidad_media'] = (df_mw['De'] + df_mw['Hasta']) / 2
    
    if 'profundidad_media' not in df_spt.columns:
        df_spt['profundidad_media'] = (df_spt['De'] + df_spt['Hasta']) / 2
    
    # Para cada ensayo MW, buscar ensayos SPT cercanos
    unique_mw_ensayos = df_mw['�tem'].unique() if '�tem' in df_mw.columns else []
    
    for mw_ensayo in unique_mw_ensayos:
        # Obtener datos del ensayo MW
        mw_data = df_mw[df_mw['�tem'] == mw_ensayo].copy()
        
        # Obtener coordenadas del ensayo MW
        mw_norte = mw_data['Norte'].iloc[0]
        mw_este = mw_data['Este'].iloc[0]
        
        # Buscar ensayos SPT cercanos
        for _, spt_row in df_spt.iterrows():
            spt_norte = spt_row['Norte']
            spt_este = spt_row['Este']
            
            # Calcular distancia euclidiana entre ensayos
            distance = np.sqrt((mw_norte - spt_norte)**2 + (mw_este - spt_este)**2)
            
            # Si están suficientemente cerca
            if distance <= max_distance:
                # Buscar muestras a profundidades similares
                for _, mw_row in mw_data.iterrows():
                    mw_prof = mw_row['profundidad_media']
                    spt_prof = spt_row['profundidad_media']
                    
                    # Si las profundidades son similares (dentro de 1 metro)
                    prof_diff = abs(mw_prof - spt_prof)
                    if prof_diff <= 1.0:
                        # Añadir par de valores correlacionados
                        correlation_data.append({
                            'profundidad': (mw_prof + spt_prof) / 2,
                            'Vs': mw_row['Vs (m/s)'],
                            'NSPT': spt_row['Nspt'],
                            'distancia_horizontal': distance,
                            'diferencia_profundidad': prof_diff,
                            'ensayo_mw': mw_ensayo,
                            'ensayo_spt': spt_row['�tem'] if '�tem' in spt_row else 'Desconocido'
                        })
    
    # Convertir a DataFrame
    df_correlation = pd.DataFrame(correlation_data)
    
    # Si no hay suficientes datos, usar método basado en profundidad
    if len(df_correlation) < 5:
        print("Datos espaciales insuficientes. Usando método basado en profundidad...")
        return create_correlation_data_by_depth(df_mw, df_spt)
    
    return df_correlation

def create_correlation_data_by_depth(df_mw, df_spt):
    """
    Método alternativo: crear datos de correlación basados solo en la profundidad.
    
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
    correlation_data = []
    
    # Verificar columnas necesarias
    if 'Vs (m/s)' not in df_mw.columns or 'Nspt' not in df_spt.columns:
        print("Faltan columnas esenciales para correlación (Vs o NSPT)")
        return pd.DataFrame()
    
    # Obtener rangos de profundidad
    if 'De' in df_mw.columns and 'Hasta' in df_mw.columns:
        df_mw['profundidad_media'] = (df_mw['De'] + df_mw['Hasta']) / 2
    
    if 'De' in df_spt.columns and 'Hasta' in df_spt.columns:
        df_spt['profundidad_media'] = (df_spt['De'] + df_spt['Hasta']) / 2
    
    if 'profundidad_media' not in df_mw.columns or 'profundidad_media' not in df_spt.columns:
        print("No se puede determinar la profundidad media")
        return pd.DataFrame()
    
    # Obtener rangos de profundidad
    min_depth = min(df_mw['profundidad_media'].min(), df_spt['profundidad_media'].min())
    max_depth = min(df_mw['profundidad_media'].max(), df_spt['profundidad_media'].max())
    
    # Crear intervalos de profundidad
    depth_intervals = np.linspace(min_depth, max_depth, 20)
    
    for depth in depth_intervals:
        # Filtrar datos cercanos a esta profundidad (dentro de 0.5 metros)
        mw_near = df_mw[(df_mw['profundidad_media'] >= depth - 0.5) & 
                         (df_mw['profundidad_media'] <= depth + 0.5)]
        
        spt_near = df_spt[(df_spt['profundidad_media'] >= depth - 0.5) & 
                           (df_spt['profundidad_media'] <= depth + 0.5)]
        
        if not mw_near.empty and not spt_near.empty:
            # Calcular valores promedio para esta profundidad
            avg_vs = mw_near['Vs (m/s)'].mean()
            avg_nspt = spt_near['Nspt'].mean()
            
            if not pd.isna(avg_vs) and not pd.isna(avg_nspt):
                correlation_data.append({
                    'profundidad': depth,
                    'Vs': avg_vs,
                    'NSPT': avg_nspt,
                    'distancia_horizontal': np.nan,
                    'diferencia_profundidad': np.nan,
                    'ensayo_mw': 'Promedio',
                    'ensayo_spt': 'Promedio'
                })
    
    return pd.DataFrame(correlation_data)

def power_law_model(x, a, b):
    """
    Modelo de ley de potencia: Vs = a * NSPT^b
    
    Parameters:
    -----------
    x : array
        Variable independiente (NSPT)
    a, b : float
        Parámetros del modelo
        
    Returns:
    --------
    array
        Predicción de Vs
    """
    return a * np.power(x, b)

def train_correlation_models(corr_data):
    """
    Entrena múltiples modelos de correlación entre Vs y NSPT.
    
    Parameters:
    -----------
    corr_data : pandas.DataFrame
        DataFrame con datos para correlación
        
    Returns:
    --------
    dict
        Diccionario con resultados de diferentes modelos
    """
    # Verificar datos suficientes
    if len(corr_data) < 5:
        print("Datos insuficientes para análisis de correlación")
        return None
    
    # Datos para modelado
    X = corr_data['NSPT'].values.reshape(-1, 1)
    y = corr_data['Vs'].values
    
    # Inicializar resultados
    results = {}
    
    # 1. Regresión Lineal Simple
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    
    results['linear'] = {
        'model': linear_model,
        'r2': r2_score(y, y_pred_linear),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_linear)),
        'mae': mean_absolute_error(y, y_pred_linear),
        'equation': f"Vs = {linear_model.coef_[0]:.2f} × NSPT + {linear_model.intercept_:.2f}",
        'predictions': y_pred_linear
    }
    
    # 2. Regresión Polinómica (grado 2)
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)
    
    # Obtener coeficientes para la ecuación
    poly_coefs = poly_model.named_steps['linearregression'].coef_
    poly_intercept = poly_model.named_steps['linearregression'].intercept_
    
    results['polynomial'] = {
        'model': poly_model,
        'r2': r2_score(y, y_pred_poly),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_poly)),
        'mae': mean_absolute_error(y, y_pred_poly),
        'equation': f"Vs = {poly_coefs[2]:.2f} × NSPT² + {poly_coefs[1]:.2f} × NSPT + {poly_intercept:.2f}",
        'predictions': y_pred_poly
    }
    
    # 3. Regresión Robusta (RANSAC)
    ransac_model = RANSACRegressor(random_state=42)
    ransac_model.fit(X, y)
    y_pred_ransac = ransac_model.predict(X)
    
    # Obtener coeficientes
    ransac_coef = ransac_model.estimator_.coef_[0]
    ransac_intercept = ransac_model.estimator_.intercept_
    
    results['ransac'] = {
        'model': ransac_model,
        'r2': r2_score(y, y_pred_ransac),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_ransac)),
        'mae': mean_absolute_error(y, y_pred_ransac),
        'equation': f"Vs = {ransac_coef:.2f} × NSPT + {ransac_intercept:.2f}",
        'predictions': y_pred_ransac
    }
    
    # 4. Regresión Robusta (Huber)
    huber_model = HuberRegressor()
    huber_model.fit(X, y)
    y_pred_huber = huber_model.predict(X)
    
    results['huber'] = {
        'model': huber_model,
        'r2': r2_score(y, y_pred_huber),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_huber)),
        'mae': mean_absolute_error(y, y_pred_huber),
        'equation': f"Vs = {huber_model.coef_[0]:.2f} × NSPT + {huber_model.intercept_:.2f}",
        'predictions': y_pred_huber
    }
    
    # 5. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    
    results['random_forest'] = {
        'model': rf_model,
        'r2': r2_score(y, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_rf)),
        'mae': mean_absolute_error(y, y_pred_rf),
        'equation': "Vs = RandomForest(NSPT)",  # Modelo no paramétrico
        'predictions': y_pred_rf
    }
    
    # 6. Ley de Potencia (Vs = a * NSPT^b) - común en correlaciones geotécnicas
    try:
        # Filtrar valores no positivos para evitar errores con logaritmos
        valid_idx = (X.flatten() > 0) & (y > 0)
        
        if np.sum(valid_idx) >= 3:  # Necesitamos al menos 3 puntos
            X_valid = X.flatten()[valid_idx]
            y_valid = y[valid_idx]
            
            # Ajustar modelo
            params, _ = curve_fit(power_law_model, X_valid, y_valid, p0=[100, 0.5])
            a, b = params
            
            # Predecir
            y_pred_power = power_law_model(X.flatten(), a, b)
            
            # Calcular métricas
            results['power_law'] = {
                'model': {'function': power_law_model, 'params': params},
                'r2': r2_score(y, y_pred_power),
                'rmse': np.sqrt(mean_squared_error(y, y_pred_power)),
                'mae': mean_absolute_error(y, y_pred_power),
                'equation': f"Vs = {a:.2f} × NSPT^{b:.3f}",
                'predictions': y_pred_power
            }
        else:
            print("Datos insuficientes para modelo de ley de potencia")
    except Exception as e:
        print(f"Error ajustando modelo de ley de potencia: {e}")
    
    # Determinar el mejor modelo basado en R²
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    results['best_model'] = best_model[0]
    
    return results

def visualize_correlation_models(corr_data, results):
    """
    Visualiza los resultados de los modelos de correlación.
    
    Parameters:
    -----------
    corr_data : pandas.DataFrame
        DataFrame con datos para correlación
    results : dict
        Diccionario con resultados de diferentes modelos
    """
    # Crear figura
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Gráfico principal: Vs vs NSPT con todos los modelos
    ax1 = fig.add_subplot(gs[0, :])
    
    # Datos originales
    scatter = ax1.scatter(
        corr_data['NSPT'],
        corr_data['Vs'],
        s=80,
        alpha=0.7,
        c=corr_data['profundidad'] if 'profundidad' in corr_data else 'blue',
        cmap='viridis',
        label='Datos Medidos'
    )
    
    # Añadir colorbar si hay datos de profundidad
    if 'profundidad' in corr_data:
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Profundidad (m)')
    
    # Ordenar valores de NSPT para graficar líneas suaves
    X_sort = np.sort(corr_data['NSPT'].values)
    X_sort = X_sort.reshape(-1, 1)
    
    # Graficar cada modelo
    colors = {
        'linear': 'red',
        'polynomial': 'green',
        'ransac': 'purple',
        'huber': 'orange',
        'random_forest': 'brown',
        'power_law': 'black'
    }
    
    for model_name, model_results in results.items():
        if model_name == 'best_model':
            continue
        
        # Obtener predicciones ordenadas
        if model_name == 'power_law':
            y_pred_sort = power_law_model(X_sort.flatten(), *model_results['model']['params'])
        elif model_name == 'random_forest':
            y_pred_sort = model_results['model'].predict(X_sort)
        elif model_name == 'polynomial':
            y_pred_sort = model_results['model'].predict(X_sort)
        else:
            y_pred_sort = model_results['model'].predict(X_sort)
        
        # Graficar modelo
        ax1.plot(
            X_sort,
            y_pred_sort,
            '-',
            color=colors.get(model_name, 'gray'),
            linewidth=2,
            label=f"{model_name.replace('_', ' ').title()}: {model_results['equation']}"
        )
    
    # Destacar el mejor modelo
    best_model_name = results['best_model']
    if best_model_name in results:
        # Obtener predicciones
        if best_model_name == 'power_law':
            y_pred_best = power_law_model(X_sort.flatten(), *results[best_model_name]['model']['params'])
        elif best_model_name == 'random_forest':
            y_pred_best = results[best_model_name]['model'].predict(X_sort)
        elif best_model_name == 'polynomial':
            y_pred_best = results[best_model_name]['model'].predict(X_sort)
        else:
            y_pred_best = results[best_model_name]['model'].predict(X_sort)
        
        # Graficar mejor modelo con línea más gruesa
        ax1.plot(
            X_sort,
            y_pred_best,
            '--',
            color=colors.get(best_model_name, 'blue'),
            linewidth=4,
            label=f"Mejor Modelo ({best_model_name.replace('_', ' ').title()})"
        )
    
    # Añadir texto con resultados del mejor modelo
    if best_model_name in results:
        best_r2 = results[best_model_name]['r2']
        best_rmse = results[best_model_name]['rmse']
        best_equation = results[best_model_name]['equation']
        
        text = f"Mejor Modelo: {best_model_name.replace('_', ' ').title()}\n"
        text += f"Ecuación: {best_equation}\n"
        text += f"R² = {best_r2:.3f}, RMSE = {best_rmse:.2f} m/s"
        
        ax1.text(
            0.05, 0.95, 
            text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax1.set_title('Modelos de Correlación entre Velocidad S y NSPT', fontsize=16)
    ax1.set_xlabel('NSPT (número de golpes)', fontsize=12)
    ax1.set_ylabel('Velocidad S (m/s)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Gráfico inferior izquierdo: Comparación de R²
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Extraer R² y RMSE para cada modelo
    model_names = [name for name in results if name != 'best_model']
    r2_values = [results[name]['r2'] for name in model_names]
    
    # Ordenar por R²
    sorted_indices = np.argsort(r2_values)[::-1]  # Descendente
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]
    
    # Crear barplot
    bars = ax2.bar(
        range(len(sorted_names)),
        sorted_r2,
        color=[colors.get(name, 'gray') for name in sorted_names]
    )
    
    # Añadir valores
    for bar, value in zip(bars, sorted_r2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax2.set_xticks(range(len(sorted_names)))
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in sorted_names], rotation=45, ha='right')
    ax2.set_title('Comparación de R² por Modelo', fontsize=12)
    ax2.set_ylabel('R²', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico inferior derecho: Comparación de RMSE
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Extraer RMSE para cada modelo
    rmse_values = [results[name]['rmse'] for name in model_names]
    
    # Ordenar por RMSE (ascendente)
    sorted_indices = np.argsort(rmse_values)
    sorted_names_rmse = [model_names[i] for i in sorted_indices]
    sorted_rmse = [rmse_values[i] for i in sorted_indices]
    
    # Crear barplot
    bars = ax3.bar(
        range(len(sorted_names_rmse)),
        sorted_rmse,
        color=[colors.get(name, 'gray') for name in sorted_names_rmse]
    )
    
    # Añadir valores
    for bar, value in zip(bars, sorted_rmse):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax3.set_xticks(range(len(sorted_names_rmse)))
    ax3.set_xticklabels([name.replace('_', ' ').title() for name in sorted_names_rmse], rotation=45, ha='right')
    ax3.set_title('Comparación de RMSE por Modelo (m/s)', fontsize=12)
    ax3.set_ylabel('RMSE (m/s)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico inferior: Residuales del mejor modelo
    ax4 = fig.add_subplot(gs[2, :])
    
    # Residuales del mejor modelo
    if best_model_name in results:
        residuals = corr_data['Vs'].values - results[best_model_name]['predictions']
        
        # Gráfico de residuales
        ax4.scatter(
            corr_data['NSPT'],
            residuals,
            s=60,
            alpha=0.7,
            c=colors.get(best_model_name, 'blue')
        )
        
        # Línea horizontal en y=0
        ax4.axhline(y=0, color='r', linestyle='-')
        
        ax4.set_title(f'Residuales del Modelo {best_model_name.replace("_", " ").title()}', fontsize=12)
        ax4.set_xlabel('NSPT (número de golpes)', fontsize=10)
        ax4.set_ylabel('Residual (m/s)', fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "correlations"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / "modelos_correlacion_vs_nspt.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figura guardada en {results_dir / 'modelos_correlacion_vs_nspt.png'}")

def save_correlation_results(corr_data, results):
    """
    Guarda los resultados de los modelos de correlación.
    
    Parameters:
    -----------
    corr_data : pandas.DataFrame
        DataFrame con datos para correlación
    results : dict
        Diccionario con resultados de diferentes modelos
    """
    # Crear directorio para guardar resultados
    results_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar datos de correlación
    corr_data.to_csv(results_dir / "datos_correlacion_vs_nspt.csv", index=False)
    
    # Crear tabla comparativa de modelos
    model_comparison = []
    
    for model_name, model_results in results.items():
        if model_name == 'best_model':
            continue
        
        model_comparison.append({
            'modelo': model_name,
            'ecuacion': model_results['equation'],
            'r2': model_results['r2'],
            'rmse': model_results['rmse'],
            'mae': model_results['mae']
        })
    
    # Convertir a DataFrame y ordenar por R²
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df = comparison_df.sort_values('r2', ascending=False)
    
    # Guardar comparación
    comparison_df.to_csv(results_dir / "comparacion_modelos_vs_nspt.csv", index=False)
    
    # Guardar resultados en un archivo de texto
    with open(results_dir / "resultados_correlacion_vs_nspt.txt", 'w') as f:
        f.write(f"Correlación Vs-NSPT\n")
        f.write(f"=================\n\n")
        
        f.write(f"Número de puntos: {len(corr_data)}\n\n")
        
        f.write(f"Comparación de Modelos:\n")
        for index, row in comparison_df.iterrows():
            f.write(f"\n{row['modelo'].replace('_', ' ').title()}:\n")
            f.write(f"  Ecuación: {row['ecuacion']}\n")
            f.write(f"  R²: {row['r2']:.3f}\n")
            f.write(f"  RMSE: {row['rmse']:.2f} m/s\n")
            f.write(f"  MAE: {row['mae']:.2f} m/s\n")
        
        f.write(f"\nMejor Modelo: {results['best_model'].replace('_', ' ').title()}\n")
    
    print(f"Resultados guardados en {results_dir}")

def main():
    """Función principal"""
    # Cargar datos procesados
    df = load_processed_data()
    
    if df is not None:
        print("Analizando correlación Vs-NSPT...")
        
        # Dividir por tipo de ensayo
        df_mw = df[df['tipo_ensayo'] == 'MW'].copy()
        df_spt = df[df['tipo_ensayo'] == 'SPT'].copy()
        
        # Verificar si hay suficientes datos para correlacionar Vs y NSPT
        if not df_mw.empty and not df_spt.empty:
            # Crear datos de correlación con análisis espacial
            corr_data = create_correlation_data(df_mw, df_spt)
            
            if len(corr_data) > 0:
                print(f"Se encontraron {len(corr_data)} pares de datos para correlación Vs-NSPT")
                
                # Entrenar modelos de correlación
                results = train_correlation_models(corr_data)
                
                if results is not None:
                    # Visualizar modelos
                    visualize_correlation_models(corr_data, results)
                    
                    # Guardar resultados
                    save_correlation_results(corr_data, results)
                    
                    print("Análisis de correlación Vs-NSPT completado con éxito!")
                else:
                    print("No se pudo completar el análisis de correlación.")
            else:
                print("No hay suficientes datos para correlacionar Vs y NSPT.")
        else:
            print("No hay suficientes datos de ensayos MW o SPT para análisis de correlación.")
    else:
        print("No se pudo cargar los datos procesados.")

if __name__ == "__main__":
    main()