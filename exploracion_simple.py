# Crear un script exploracion_simple.py en la raíz del proyecto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cargar datos
csv_path = Path("data/raw/Colegio1MariaIgnacia.csv")
df = pd.read_csv(csv_path, delimiter=';', encoding='latin1')

# Ver primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Dimensiones del dataset
print(f"\nDimensiones: {df.shape}")

# Información sobre columnas
print("\nInformación sobre columnas:")
print(df.info())

# Estadísticas básicas
print("\nEstadísticas básicas:")
print(df.describe())

# Ver tipos de ensayos disponibles
ensayos = df['�tem'].str.split('-').str[0].unique()
print(f"\nTipos de ensayos: {ensayos}")

# Guardar resultados
output_dir = Path("results/exploracion_inicial")
output_dir.mkdir(exist_ok=True, parents=True)

# Crear un resumen
with open(output_dir / "resumen_datos.txt", "w") as f:
    f.write(f"Archivo: {csv_path}\n")
    f.write(f"Dimensiones: {df.shape}\n")
    f.write(f"Tipos de ensayos: {ensayos}\n")
    
print(f"\nResumen guardado en {output_dir / 'resumen_datos.txt'}")