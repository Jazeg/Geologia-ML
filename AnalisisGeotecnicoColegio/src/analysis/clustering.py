#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análisis de clustering de propiedades geotécnicas.
Identifica grupos naturales de suelos basados en sus propiedades.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

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

def prepare_data_for_clustering(df):
    """
    Prepara los datos para análisis de clustering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
        
    Returns:
    --------
    tuple
        (DataFrame con datos para clustering, lista de columnas utilizadas)
    """
    # Seleccionar propiedades relevantes para clustering
    potential_features = [
        'profundidad_media', 'Vs (m/s)', 'Nspt', '(N1)60', 
        'LL %', 'LP', 'IP %', 'W%', 'Gravas', 'Arenas', 'Finos'
    ]
    
    # Filtrar columnas que existen en el DataFrame
    features = [col for col in potential_features if col in df.columns]
    
    # Crear DataFrame con solo las características relevantes
    clustering_data = df[features].copy()
    
    # Eliminar filas con valores NaN
    clustering_data = clustering_data.dropna()
    
    print(f"Datos preparados para clustering: {clustering_data.shape[0]} muestras con {clustering_data.shape[1]} características")
    
    return clustering_data, features

def determine_optimal_clusters(data, max_clusters=10):
    """
    Determina el número óptimo de clusters usando el método del codo y el score de silueta.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con datos preparados para clustering
    max_clusters : int, optional
        Número máximo de clusters a evaluar
        
    Returns:
    --------
    int
        Número óptimo de clusters
    """
    # Escalar datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Calcular inercia y score de silueta para diferentes números de clusters
    inertias = []
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # Entrenar modelo K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Guardar inercia
        inertias.append(kmeans.inertia_)
        
        # Calcular score de silueta
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"Para n_clusters = {n_clusters}, el score de silueta es: {silhouette_avg:.3f}")
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de inercia (método del codo)
    ax1.plot(range(2, max_clusters + 1), inertias, 'bo-')
    ax1.set_xlabel('Número de clusters')
    ax1.set_ylabel('Inercia')
    ax1.set_title('Método del Codo')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de score de silueta
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    ax2.set_xlabel('Número de clusters')
    ax2.set_ylabel('Score de Silueta')
    ax2.set_title('Score de Silueta vs. Número de Clusters')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "clustering"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / "optimal_clusters.png", dpi=300, bbox_inches='tight')
    
    # Determinar número óptimo de clusters
    # Usamos el valor con mayor score de silueta
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 porque empezamos desde 2 clusters
    
    print(f"Número óptimo de clusters basado en score de silueta: {optimal_clusters}")
    
    return optimal_clusters

def perform_kmeans_clustering(data, n_clusters):
    """
    Realiza clustering K-means.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con datos preparados para clustering
    n_clusters : int
        Número de clusters
        
    Returns:
    --------
    tuple
        (modelo KMeans entrenado, datos con etiquetas de cluster)
    """
    # Escalar datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Entrenar modelo K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Añadir etiquetas de cluster al DataFrame
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels
    
    return kmeans, data_with_clusters, scaler

def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Realiza clustering DBSCAN.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame con datos preparados para clustering
    eps : float, optional
        Distancia máxima entre dos muestras para ser consideradas en el mismo vecindario
    min_samples : int, optional
        Número mínimo de muestras en un vecindario para que un punto sea considerado core point
        
    Returns:
    --------
    tuple
        (modelo DBSCAN entrenado, datos con etiquetas de cluster)
    """
    # Escalar datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Entrenar modelo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(scaled_data)
    
    # Añadir etiquetas de cluster al DataFrame
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels
    
    return dbscan, data_with_clusters, scaler

def visualize_clusters_2d(data_with_clusters, features, method_name="KMeans"):
    """
    Visualiza los clusters en 2D usando PCA.
    
    Parameters:
    -----------
    data_with_clusters : pandas.DataFrame
        DataFrame con datos y etiquetas de cluster
    features : list
        Lista de nombres de características utilizadas
    method_name : str, optional
        Nombre del método de clustering
    """
    # Obtener datos sin la columna de cluster
    X = data_with_clusters[features]
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = data_with_clusters['cluster']
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar clusters
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='cluster',
        palette='viridis',
        data=pca_df,
        s=80,
        alpha=0.7
    )
    
    # Añadir título y etiquetas
    plt.title(f'Visualización de Clusters con {method_name} (PCA 2D)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza explicada)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza explicada)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "clustering"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / f"clusters_2d_{method_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # También crear un gráfico de pares para visualizar relaciones entre características originales
    if len(features) <= 6:  # Limitar para no crear un gráfico demasiado grande
        plt.figure(figsize=(15, 15))
        
        # Añadir cluster a los datos
        plot_data = data_with_clusters[features + ['cluster']].copy()
        
        # Crear gráfico de pares
        sns.pairplot(
            plot_data,
            hue='cluster',
            palette='viridis',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 60}
        )
        
        plt.suptitle(f'Relaciones entre Características por Cluster ({method_name})', fontsize=16, y=1.02)
        
        # Guardar figura
        plt.savefig(results_dir / f"clusters_pairs_{method_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualización de clusters con {method_name} guardada.")

def visualize_clusters_3d(data_with_clusters, features, method_name="KMeans"):
    """
    Visualiza los clusters en 3D usando PCA.
    
    Parameters:
    -----------
    data_with_clusters : pandas.DataFrame
        DataFrame con datos y etiquetas de cluster
    features : list
        Lista de nombres de características utilizadas
    method_name : str, optional
        Nombre del método de clustering
    """
    # Verificar que hay suficientes características
    if len(features) < 3:
        print("Se necesitan al menos 3 características para visualización 3D")
        return
    
    # Obtener datos sin la columna de cluster
    X = data_with_clusters[features]
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA para reducir a 3 dimensiones
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2', 'PC3']
    )
    pca_df['cluster'] = data_with_clusters['cluster']
    
    # Crear figura 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar cada cluster
    for cluster in pca_df['cluster'].unique():
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            cluster_data['PC3'],
            label=f'Cluster {cluster}',
            s=60,
            alpha=0.7
        )
    
    # Añadir título y etiquetas
    ax.set_title(f'Visualización 3D de Clusters con {method_name}', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=12)
    ax.legend()
    
    # Crear directorio para guardar la figura
    results_dir = Path(__file__).resolve().parents[2] / "results" / "figures" / "clustering"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(results_dir / f"clusters_3d_{method_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualización 3D de clusters con {method_name} guardada.")

def analyze_cluster_properties(data_with_clusters, features):
    """
    Analiza las propiedades de cada cluster.
    
    Parameters:
    -----------
    data_with_clusters : pandas.DataFrame
        DataFrame con datos y etiquetas de cluster
    features : list
        Lista de nombres de características utilizadas
    """
    # Obtener estadísticas por cluster
    cluster_stats = data_with_clusters.groupby('cluster')[features].agg(
        ['mean', 'std', 'min', 'max', 'count']
    )
    
    # Crear directorio para guardar resultados
    results_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar estadísticas
    cluster_stats.to_csv(results_dir / "cluster_statistics.csv")
    
    print(f"Estadísticas de clusters guardadas en {results_dir / 'cluster_statistics.csv'}")
    
    # Visualizar características promedio por cluster
    plt.figure(figsize=(14, 10))
    
    # Preparar datos para visualización
    cluster_means = data_with_clusters.groupby('cluster')[features].mean()
    
    # Normalizar para comparación
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # Crear mapa de calor
    sns.heatmap(
        cluster_means_scaled,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Características Promedio por Cluster (Normalizadas)', fontsize=16)
    plt.ylabel('Cluster', fontsize=12)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(results_dir / "../figures/clustering/cluster_profiles.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualizar distribución de profundidades por cluster
    if 'profundidad_media' in features:
        plt.figure(figsize=(12, 8))
        
        sns.boxplot(
            x='cluster',
            y='profundidad_media',
            data=data_with_clusters,
            palette='viridis'
        )
        
        plt.title('Distribución de Profundidades por Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Profundidad Media (m)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(results_dir / "../figures/clustering/depth_by_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Visualizar distribución de Vs y NSPT por cluster
    important_props = ['Vs (m/s)', 'Nspt']
    existing_props = [prop for prop in important_props if prop in features]
    
    if existing_props:
        fig, axs = plt.subplots(len(existing_props), 1, figsize=(12, 6*len(existing_props)))
        
        if len(existing_props) == 1:
            axs = [axs]
        
        for i, prop in enumerate(existing_props):
            sns.boxplot(
                x='cluster',
                y=prop,
                data=data_with_clusters,
                palette='viridis',
                ax=axs[i]
            )
            
            axs[i].set_title(f'Distribución de {prop} por Cluster', fontsize=14)
            axs[i].set_xlabel('Cluster', fontsize=12)
            axs[i].set_ylabel(prop, fontsize=12)
            axs[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(results_dir / "../figures/clustering/properties_by_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Análisis de propiedades por cluster completado.")

def main():
    """Función principal"""
    # Cargar datos procesados
    df = load_processed_data()
    
    if df is not None:
        print("Realizando análisis de clustering...")
        
        # Preparar datos para clustering
        clustering_data, features = prepare_data_for_clustering(df)
        
        if len(clustering_data) > 0:
            # Determinar número óptimo de clusters
            n_clusters = determine_optimal_clusters(clustering_data)
            
            # Realizar clustering K-means
            print(f"\nRealizando clustering K-means con {n_clusters} clusters...")
            kmeans_model, kmeans_data, kmeans_scaler = perform_kmeans_clustering(clustering_data, n_clusters)
            
            # Visualizar resultados K-means
            visualize_clusters_2d(kmeans_data, features, "KMeans")
            visualize_clusters_3d(kmeans_data, features, "KMeans")
            
            # Analizar propiedades por cluster (K-means)
            analyze_cluster_properties(kmeans_data, features)
            
            # Realizar clustering DBSCAN
            print("\nRealizando clustering DBSCAN...")
            # Buscar parámetros adecuados para DBSCAN
            # En un caso real, se requeriría una búsqueda más exhaustiva
            eps = 0.8  # Distancia para considerar puntos como vecinos
            min_samples = max(5, int(len(clustering_data) * 0.05))  # Al menos 5% de los datos
            
            dbscan_model, dbscan_data, dbscan_scaler = perform_dbscan_clustering(
                clustering_data, eps=eps, min_samples=min_samples
            )
            
            # Contar número de clusters (excluyendo ruido, que es -1)
            n_dbscan_clusters = len(set(dbscan_data['cluster'])) - (1 if -1 in dbscan_data['cluster'].values else 0)
            print(f"DBSCAN encontró {n_dbscan_clusters} clusters y {(dbscan_data['cluster'] == -1).sum()} puntos de ruido")
            
            if n_dbscan_clusters > 1:
                # Visualizar resultados DBSCAN
                visualize_clusters_2d(dbscan_data, features, "DBSCAN")
                visualize_clusters_3d(dbscan_data, features, "DBSCAN")
                
                # Analizar propiedades por cluster (DBSCAN)
                # Excluir puntos de ruido
                dbscan_data_no_noise = dbscan_data[dbscan_data['cluster'] != -1].copy()
                if len(dbscan_data_no_noise) > 0:
                    analyze_cluster_properties(dbscan_data_no_noise, features)
            else:
                print("DBSCAN no encontró clusters significativos. Ajusta los parámetros.")
            
            # Guardar resultados
            results_dir = Path(__file__).resolve().parents[2] / "results" / "predictions"
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Guardar datos con etiquetas de cluster
            kmeans_data.to_csv(results_dir / "kmeans_clusters.csv", index=False)
            dbscan_data.to_csv(results_dir / "dbscan_clusters.csv", index=False)
            
            print("Análisis de clustering completado con éxito!")
        else:
            print("No hay suficientes datos para realizar clustering.")
    else:
        print("No se pudo cargar los datos procesados.")

if __name__ == "__main__":
    main()