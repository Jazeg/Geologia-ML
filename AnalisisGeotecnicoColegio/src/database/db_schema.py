#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Definición del esquema de base de datos para datos geotécnicos.
"""

from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Proyecto(Base):
    """
    Tabla para almacenar información de proyectos (colegios, edificios, etc.)
    """
    __tablename__ = 'proyectos'
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(200), nullable=False)
    ubicacion = Column(String(200), nullable=False)
    ciudad = Column(String(100), nullable=True)
    provincia = Column(String(100), nullable=True)
    fecha_inicio = Column(Date, nullable=True)
    descripcion = Column(Text, nullable=True)
    
    # Relaciones
    ensayos = relationship("Ensayo", back_populates="proyecto")
    
    def __repr__(self):
        return f"<Proyecto(nombre='{self.nombre}', ubicacion='{self.ubicacion}')>"

class Ensayo(Base):
    """
    Tabla para almacenar información de ensayos geotécnicos
    """
    __tablename__ = 'ensayos'
    
    id = Column(Integer, primary_key=True)
    codigo = Column(String(50), nullable=False)  # SPT-1, C-1, MW-1, etc.
    tipo = Column(String(20), nullable=False)    # SPT, C, MW
    norte = Column(Float, nullable=False)
    este = Column(Float, nullable=False)
    fecha_ensayo = Column(Date, nullable=True)
    observaciones = Column(Text, nullable=True)
    
    # Claves foráneas
    proyecto_id = Column(Integer, ForeignKey('proyectos.id'))
    
    # Relaciones
    proyecto = relationship("Proyecto", back_populates="ensayos")
    muestras = relationship("Muestra", back_populates="ensayo")
    
    def __repr__(self):
        return f"<Ensayo(codigo='{self.codigo}', tipo='{self.tipo}')>"

class Muestra(Base):
    """
    Tabla para almacenar información de muestras extraídas en cada ensayo
    """
    __tablename__ = 'muestras'
    
    id = Column(Integer, primary_key=True)
    codigo = Column(String(50), nullable=False)  # M-1, M-2, etc.
    profundidad_inicial = Column(Float, nullable=False)
    profundidad_final = Column(Float, nullable=False)
    potencia = Column(Float, nullable=False)
    
    # Claves foráneas
    ensayo_id = Column(Integer, ForeignKey('ensayos.id'))
    
    # Relaciones
    ensayo = relationship("Ensayo", back_populates="muestras")
    propiedades = relationship("PropiedadesMuestra", uselist=False, back_populates="muestra")
    
    def __repr__(self):
        return f"<Muestra(codigo='{self.codigo}', prof='{self.profundidad_inicial}-{self.profundidad_final}m')>"

class PropiedadesMuestra(Base):
    """
    Tabla para almacenar propiedades físicas de cada muestra
    """
    __tablename__ = 'propiedades_muestras'
    
    id = Column(Integer, primary_key=True)
    
    # Propiedades sísmicas
    vs = Column(Float, nullable=True)  # Velocidad S (m/s)
    vp = Column(Float, nullable=True)  # Velocidad P (m/s)
    
    # Propiedades SPT
    nspt = Column(Float, nullable=True)
    n160 = Column(Float, nullable=True)
    
    # Clasificación
    sucs = Column(String(10), nullable=True)
    
    # Límites de Atterberg
    ll = Column(Float, nullable=True)  # Límite líquido
    lp = Column(Float, nullable=True)  # Límite plástico
    ip = Column(Float, nullable=True)  # Índice de plasticidad
    
    # Otras propiedades físicas
    w = Column(Float, nullable=True)    # Humedad (%)
    phi = Column(Float, nullable=True)  # Ángulo de fricción
    c = Column(Float, nullable=True)    # Cohesión
    densidad_humeda = Column(Float, nullable=True)  # Densidad húmeda (gr/cm3)
    densidad_seca = Column(Float, nullable=True)    # Densidad seca (gr/cm3)
    
    # Capacidad de carga
    qadm = Column(Float, nullable=True)  # Capacidad admisible (kg/cm2)
    qult = Column(Float, nullable=True)  # Capacidad última (kg/cm2)
    
    # Composición granulométrica
    gravas = Column(Float, nullable=True)  # Porcentaje de gravas
    arenas = Column(Float, nullable=True)  # Porcentaje de arenas
    finos = Column(Float, nullable=True)   # Porcentaje de finos
    
    # Propiedades químicas
    sst = Column(Float, nullable=True)  # SST (ppm)
    so4 = Column(Float, nullable=True)  # SO4 (ppm)
    cl = Column(Float, nullable=True)   # CL (ppm)
    ph = Column(Float, nullable=True)   # pH
    
    # Otras propiedades
    tipo_suelo = Column(String(10), nullable=True)
    colapso = Column(String(20), nullable=True)
    
    # Claves foráneas
    muestra_id = Column(Integer, ForeignKey('muestras.id'))
    
    # Relaciones
    muestra = relationship("Muestra", back_populates="propiedades")
    
    def __repr__(self):
        return f"<PropiedadesMuestra(muestra_id={self.muestra_id})>"