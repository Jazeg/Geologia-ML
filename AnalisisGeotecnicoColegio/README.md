# Análisis Geotécnico - Colegio María Ignacia

## Descripción del Proyecto

Este proyecto organiza, analiza y predice propiedades geotécnicas de suelos basados en ensayos de campo realizados en el Colegio María Ignacia. Se utilizan técnicas de Machine Learning para encontrar correlaciones entre propiedades físicas (especialmente Velocidad S y NSPT) y predecir características a profundidades mayores que las muestreadas.

## Estructura del Proyecto
AnalisisGeotecnicoColegio/
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos sin procesar
│   ├── processed/                 # Datos procesados
│   └── interim/                   # Datos intermedios por tipo de ensayo
├── src/                           # Código fuente
│   ├── data_processing/           # Scripts para procesar datos
│   ├── database/                  # Código para la base de datos
│   ├── analysis/                  # Scripts para análisis
│   ├── modeling/                  # Scripts para modelos predictivos
│   └── visualization/             # Scripts para visualización
├── results/                       # Resultados generados
│   ├── figures/                   # Gráficos generados
│   │   ├── correlations/          # Gráficos de correlaciones
│   │   ├── profiles/              # Perfiles de propiedades vs profundidad
│   │   └── maps/                  # Mapas y visualizaciones espaciales
│   ├── tables/                    # Tablas de resultados
│   └── predictions/               # Datos de predicciones
├── models/                        # Modelos entrenados
│   ├── trained/                   # Modelos guardados
│   └── validation/                # Validación de modelos
└── notebooks/                     # Jupyter notebooks para análisis

## Datos

El conjunto de datos incluye información de diferentes tipos de ensayos geotécnicos:

1. **SPT (Ensayo de Penetración Estándar)**
   - NSPT y (N1)60
   - Propiedades físicas (límites de Atterberg, humedad)
   - Clasificación SUCS
   - Granulometría

2. **Calicatas**
   - Descripción de estratos
   - Propiedades físicas
   - Clasificación SUCS
   - Granulometría

3. **Ensayos MW (Microtremores)**
   - Velocidad de ondas sísmicas (Vs)
   - Perfiles a mayor profundidad (hasta 36+ metros)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Jazeg/AnalisisGeotecnicoColegio.git
cd AnalisisGeotecnicoColegio

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar datos
# (Copiar Colegio1MariaIgnacia.csv en data/raw/)
