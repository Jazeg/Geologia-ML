#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Operaciones de base de datos para datos geotécnicos.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from .db_schema import Base, Proyecto, Ensayo, Muestra, PropiedadesMuestra

class GeotechDatabaseManager:
    """
    Clase para gestionar operaciones de base de datos para datos geotécnicos.
    """
    
    def __init__(self, db_url=None):
        """
        Inicializa el administrador de base de datos.
        
        Parameters:
        -----------
        db_url : str, optional
            URL de conexión a la base de datos
            Formato: 'postgresql://usuario:contraseña@host:puerto/nombre_db'
            Si no se proporciona, se intentará usar la variable de entorno DATABASE_URL
        """
        if db_url is None:
            db_url = os.environ.get('DATABASE_URL')
            
        if db_url is None:
            # URL por defecto para desarrollo local
            db_url = 'postgresql://postgres:postgres@localhost:5432/geotech_data'
            
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """
        Crea todas las tablas definidas en el esquema.
        """
        Base.metadata.create_all(self.engine)
        print("Tablas creadas con éxito.")
        
    def drop_tables(self):
        """
        Elimina todas las tablas definidas en el esquema.
        """
        Base.metadata.drop_all(self.engine)
        print("Tablas eliminadas con éxito.")
        
    def import_from_csv(self, csv_path, proyecto_nombre=None):
        """
        Importa datos desde un archivo CSV a la base de datos.
        
        Parameters:
        -----------
        csv_path : str
            Ruta al archivo CSV
        proyecto_nombre : str, optional
            Nombre del proyecto al que pertenecen los datos.
            Si no se proporciona, se usará el nombre del archivo.
        """
        # Cargar datos del CSV
        try:
            # Probar diferentes codificaciones
            encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, delimiter=';', encoding=encoding)
                    df = df.dropna(how='all')
                    if len(df.columns) > 1 and len(df) > 0:
                        print(f"Archivo cargado con éxito usando codificación: {encoding}")
                        break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("No se pudo decodificar el archivo con las codificaciones disponibles")
        except Exception as e:
            print(f"Error al cargar el archivo CSV: {e}")
            return False
        
        # Reemplazar '-' por NaN
        df = df.replace('-', np.nan)
        
        # Convertir columnas numéricas
        numeric_cols = ['Norte', 'Este', 'De', 'Hasta', 'potencia', 
                        'Vs (m/s)', 'Vp (m/s)', 'Nspt', '(N1)60', 
                        'LL %', 'LP', 'IP %', 'W%', 'C (Kg/cm2)', 
                        '? h�meda (gr/cm3)', '? seca (gr/cm3)', 
                        'Qadm. (kg/cm2)', 'Qult. (kg/cm2)',
                        'SST (ppm)', 'SO4 (ppm)', 'CL (ppm)', 'pH (ppm)',
                        'Gravas', 'Arenas', 'Finos']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Determinar el nombre del proyecto si no se proporcionó
        if proyecto_nombre is None:
            proyecto_nombre = Path(csv_path).stem
        
        # Abrir sesión de base de datos
        session = self.Session()
        
        try:
            # Crear o recuperar proyecto
            proyecto = session.query(Proyecto).filter_by(nombre=proyecto_nombre).first()
            
            if proyecto is None:
                proyecto = Proyecto(
                    nombre=proyecto_nombre,
                    ubicacion="Por definir",
                    fecha_inicio=datetime.now().date()
                )
                session.add(proyecto)
                session.flush()  # Para obtener el ID
            
            # Procesar cada fila del DataFrame
            for _, row in df.iterrows():
                # Determinar tipo de ensayo
                ensayo_codigo = row['�tem'] if '�tem' in row else "Desconocido"
                
                if pd.isna(ensayo_codigo) or ensayo_codigo == "":
                    continue
                
                tipo_ensayo = ensayo_codigo.split('-')[0] if '-' in ensayo_codigo else "Desconocido"
                
                # Verificar si ya existe el ensayo
                ensayo = session.query(Ensayo).filter_by(
                    codigo=ensayo_codigo, 
                    proyecto_id=proyecto.id
                ).first()
                
                if ensayo is None:
                    # Crear nuevo ensayo
                    ensayo = Ensayo(
                        codigo=ensayo_codigo,
                        tipo=tipo_ensayo,
                        norte=row['Norte'] if 'Norte' in row and not pd.isna(row['Norte']) else 0,
                        este=row['Este'] if 'Este' in row and not pd.isna(row['Este']) else 0,
                        proyecto_id=proyecto.id
                    )
                    session.add(ensayo)
                    session.flush()  # Para obtener el ID
                
                # Crear muestra si tiene información de profundidad
                if 'De' in row and 'Hasta' in row and not pd.isna(row['De']) and not pd.isna(row['Hasta']):
                    muestra_codigo = row['Muestra'] if 'Muestra' in row and not pd.isna(row['Muestra']) else f"M-{int(row['De'])}-{int(row['Hasta'])}"
                    
                    # Verificar si ya existe la muestra
                    muestra = session.query(Muestra).filter_by(
                        codigo=muestra_codigo,
                        ensayo_id=ensayo.id,
                        profundidad_inicial=row['De'],
                        profundidad_final=row['Hasta']
                    ).first()
                    
                    if muestra is None:
                        # Crear nueva muestra
                        muestra = Muestra(
                            codigo=muestra_codigo,
                            profundidad_inicial=row['De'],
                            profundidad_final=row['Hasta'],
                            potencia=row['potencia'] if 'potencia' in row and not pd.isna(row['potencia']) 
                                    else row['Hasta'] - row['De'],
                            ensayo_id=ensayo.id
                        )
                        session.add(muestra)
                        session.flush()  # Para obtener el ID
                    
                    # Crear propiedades de la muestra
                    propiedades = PropiedadesMuestra(
                        muestra_id=muestra.id,
                        vs=row['Vs (m/s)'] if 'Vs (m/s)' in row else None,
                        vp=row['Vp (m/s)'] if 'Vp (m/s)' in row else None,
                        nspt=row['Nspt'] if 'Nspt' in row else None,
                        n160=row['(N1)60'] if '(N1)60' in row else None,
                        sucs=row['SUCS'] if 'SUCS' in row else None,
                        ll=row['LL %'] if 'LL %' in row else None,
                        lp=row['LP'] if 'LP' in row else None,
                        ip=row['IP %'] if 'IP %' in row else None,
                        w=row['W%'] if 'W%' in row else None,
                        densidad_humeda=row['? h�meda (gr/cm3)'] if '? h�meda (gr/cm3)' in row else None,
                        densidad_seca=row['? seca (gr/cm3)'] if '? seca (gr/cm3)' in row else None,
                        qadm=row['Qadm. (kg/cm2)'] if 'Qadm. (kg/cm2)' in row else None,
                        qult=row['Qult. (kg/cm2)'] if 'Qult. (kg/cm2)' in row else None,
                        tipo_suelo=row['Tipo de Suelo'] if 'Tipo de Suelo' in row else None,
                        gravas=row['Gravas'] if 'Gravas' in row else None,
                        arenas=row['Arenas'] if 'Arenas' in row else None,
                        finos=row['Finos'] if 'Finos' in row else None,
                        sst=row['SST (ppm)'] if 'SST (ppm)' in row else None,
                        so4=row['SO4 (ppm)'] if 'SO4 (ppm)' in row else None,
                        cl=row['CL (ppm)'] if 'CL (ppm)' in row else None,
                        ph=row['pH (ppm)'] if 'pH (ppm)' in row else None,
                        colapso=row['Colapso'] if 'Colapso' in row else None
                    )
                    
                    # Verificar si ya existen propiedades para esta muestra
                    existing_props = session.query(PropiedadesMuestra).filter_by(
                        muestra_id=muestra.id
                    ).first()
                    
                    if existing_props:
                        # Actualizar propiedades existentes
                        for key, value in propiedades.__dict__.items():
                            if not key.startswith('_') and value is not None:
                                setattr(existing_props, key, value)
                    else:
                        # Agregar nuevas propiedades
                        session.add(propiedades)
            
            # Confirmar cambios
            session.commit()
            print(f"Datos importados con éxito al proyecto '{proyecto_nombre}'")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error al importar datos: {e}")
            return False
            
        finally:
            session.close()
    
    def export_to_csv(self, output_path, proyecto_id=None):
        """
        Exporta datos de un proyecto a un archivo CSV.
        
        Parameters:
        -----------
        output_path : str
            Ruta donde se guardará el archivo CSV
        proyecto_id : int, optional
            ID del proyecto a exportar. Si no se proporciona, se exportarán todos los datos.
        """
        session = self.Session()
        
        try:
            # Consulta SQL para obtener datos completos
            query = """
            SELECT 
                p.nombre AS proyecto,
                e.codigo AS ensayo,
                e.tipo AS tipo_ensayo,
                e.norte,
                e.este,
                m.codigo AS muestra,
                m.profundidad_inicial AS de,
                m.profundidad_final AS hasta,
                m.potencia,
                pm.vs AS "Vs (m/s)",
                pm.vp AS "Vp (m/s)",
                pm.nspt AS "Nspt",
                pm.n160 AS "(N1)60",
                pm.sucs AS "SUCS",
                pm.ll AS "LL %",
                pm.lp AS "LP",
                pm.ip AS "IP %",
                pm.w AS "W%",
                pm.phi AS "φ °",
                pm.c AS "C (Kg/cm2)",
                pm.densidad_humeda AS "ρ húmeda (gr/cm3)",
                pm.densidad_seca AS "ρ seca (gr/cm3)",
                pm.qadm AS "Qadm. (kg/cm2)",
                pm.qult AS "Qult. (kg/cm2)",
                pm.tipo_suelo AS "Tipo de Suelo",
                pm.sst AS "SST (ppm)",
                pm.so4 AS "SO4 (ppm)",
                pm.cl AS "CL (ppm)",
                pm.ph AS "pH (ppm)",
                pm.colapso AS "Colapso",
                pm.gravas AS "Gravas",
                pm.arenas AS "Arenas",
                pm.finos AS "Finos"
            FROM 
                proyectos p
                JOIN ensayos e ON p.id = e.proyecto_id
                JOIN muestras m ON e.id = m.ensayo_id
                LEFT JOIN propiedades_muestras pm ON m.id = pm.muestra_id
            """
            
            if proyecto_id is not None:
                query += f" WHERE p.id = {proyecto_id}"
                
            query += " ORDER BY e.codigo, m.profundidad_inicial"
            
            # Ejecutar consulta
            df = pd.read_sql(text(query), session.bind)
            
            # Guardar a CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"Datos exportados con éxito a: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error al exportar datos: {e}")
            return False
            
        finally:
            session.close()
    
    def get_project_summary(self, proyecto_id=None):
        """
        Obtiene un resumen de los proyectos en la base de datos.
        
        Parameters:
        -----------
        proyecto_id : int, optional
            ID del proyecto específico. Si no se proporciona, se muestra resumen de todos.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame con el resumen
        """
        session = self.Session()
        
        try:
            # Consulta para contar ensayos y muestras por proyecto
            query = session.query(
                Proyecto.id,
                Proyecto.nombre,
                Proyecto.ubicacion,
                func.count(Ensayo.id).label('num_ensayos'),
                func.count(Muestra.id).label('num_muestras')
            ).outerjoin(
                Ensayo, Proyecto.id == Ensayo.proyecto_id
            ).outerjoin(
                Muestra, Ensayo.id == Muestra.ensayo_id
            ).group_by(
                Proyecto.id, Proyecto.nombre, Proyecto.ubicacion
            )
            
            if proyecto_id is not None:
                query = query.filter(Proyecto.id == proyecto_id)
                
            # Ejecutar consulta y convertir a DataFrame
            result = query.all()
            columns = ['id', 'nombre', 'ubicacion', 'num_ensayos', 'num_muestras']
            df = pd.DataFrame(result, columns=columns)
            
            return df
            
        except Exception as e:
            print(f"Error al obtener resumen de proyectos: {e}")
            return None
            
        finally:
            session.close()
            
    def get_test_summary(self, proyecto_id=None):
        """
        Obtiene un resumen de los ensayos en la base de datos.
        
        Parameters:
        -----------
        proyecto_id : int, optional
            ID del proyecto específico. Si no se proporciona, se muestran todos.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame con el resumen
        """
        session = self.Session()
        
        try:
            # Consulta para obtener detalles de ensayos
            query = session.query(
                Proyecto.nombre.label('proyecto'),
                Ensayo.codigo,
                Ensayo.tipo,
                Ensayo.norte,
                Ensayo.este,
                func.count(Muestra.id).label('num_muestras'),
                func.min(Muestra.profundidad_inicial).label('prof_min'),
                func.max(Muestra.profundidad_final).label('prof_max')
            ).join(
                Proyecto, Ensayo.proyecto_id == Proyecto.id
            ).outerjoin(
                Muestra, Ensayo.id == Muestra.ensayo_id
            ).group_by(
                Proyecto.nombre, Ensayo.codigo, Ensayo.tipo, Ensayo.norte, Ensayo.este
            )
            
            if proyecto_id is not None:
                query = query.filter(Proyecto.id == proyecto_id)
                
            # Ejecutar consulta y convertir a DataFrame
            result = query.all()
            columns = ['proyecto', 'codigo', 'tipo', 'norte', 'este', 
                      'num_muestras', 'prof_min', 'prof_max']
            df = pd.DataFrame(result, columns=columns)
            
            return df
            
        except Exception as e:
            print(f"Error al obtener resumen de ensayos: {e}")
            return None
            
        finally:
            session.close()
            
    def get_soil_properties(self, ensayo_codigo=None, proyecto_id=None):
        """
        Obtiene las propiedades de suelo de un ensayo específico o proyecto.
        
        Parameters:
        -----------
        ensayo_codigo : str, optional
            Código del ensayo específico
        proyecto_id : int, optional
            ID del proyecto específico
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame con las propiedades del suelo
        """
        session = self.Session()
        
        try:
            # Consulta para obtener propiedades
            query = """
            SELECT 
                p.nombre AS proyecto,
                e.codigo AS ensayo,
                e.tipo AS tipo_ensayo,
                m.codigo AS muestra,
                m.profundidad_inicial AS de,
                m.profundidad_final AS hasta,
                m.potencia,
                pm.vs AS "Vs (m/s)",
                pm.nspt AS "Nspt",
                pm.n160 AS "(N1)60",
                pm.sucs AS "SUCS",
                pm.ll AS "LL %",
                pm.lp AS "LP",
                pm.ip AS "IP %",
                pm.w AS "W%",
                pm.gravas AS "Gravas",
                pm.arenas AS "Arenas",
                pm.finos AS "Finos"
            FROM 
                proyectos p
                JOIN ensayos e ON p.id = e.proyecto_id
                JOIN muestras m ON e.id = m.ensayo_id
                LEFT JOIN propiedades_muestras pm ON m.id = pm.muestra_id
            WHERE 1=1
            """
            
            params = {}
            
            if ensayo_codigo is not None:
                query += " AND e.codigo = :ensayo_codigo"
                params['ensayo_codigo'] = ensayo_codigo
                
            if proyecto_id is not None:
                query += " AND p.id = :proyecto_id"
                params['proyecto_id'] = proyecto_id
                
            query += " ORDER BY e.codigo, m.profundidad_inicial"
            
            # Ejecutar consulta
            df = pd.read_sql(text(query), session.bind, params=params)
            
            return df
            
        except Exception as e:
            print(f"Error al obtener propiedades del suelo: {e}")
            return None
            
        finally:
            session.close()

# Función de ejemplo de uso
def main():
    """Función principal de ejemplo."""
    # Crear administrador de base de datos
    db_manager = GeotechDatabaseManager()
    
    # Crear tablas
    db_manager.create_tables()
    
    # Importar datos de ejemplo
    project_dir = Path(__file__).resolve().parents[2]
    csv_path = project_dir / "data" / "raw" / "Colegio1MariaIgnacia.csv"
    
    if csv_path.exists():
        db_manager.import_from_csv(csv_path, "Colegio María Ignacia")
        
        # Mostrar resumen de proyectos
        projects = db_manager.get_project_summary()
        print("\nResumen de Proyectos:")
        print(projects)
        
        # Mostrar resumen de ensayos
        tests = db_manager.get_test_summary()
        print("\nResumen de Ensayos:")
        print(tests)
    else:
        print(f"No se encontró el archivo: {csv_path}")

if __name__ == "__main__":
    main()