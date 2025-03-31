#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para configuración y preparación del entorno de trabajo.
"""

import os
import json
import logging
from pathlib import Path
from core.execution_tracker import add_to_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def setup_project_directories():
    """
    Crea la estructura de directorios necesaria para el proyecto.
    
    Returns:
        Mensaje de estado de la operación
    """
    logger.info("Configurando estructura de directorios del proyecto")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "setup_directories",
        "Creando estructura de directorios necesaria"
    )
    
    try:
        # Definir directorios a crear
        directories = [
            "documentos",
            "documentos/Referencias",
            "documentos/Propuestas",
            "propuestas"  # Para las propuestas generadas
        ]
        
        # Crear directorios
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio creado: {directory}")
            else:
                logger.info(f"Directorio ya existe: {directory}")
        
        # Registrar éxito
        add_to_execution_path(
            "setup_directories_result",
            "Estructura de directorios creada exitosamente"
        )
        
        return "Estructura de directorios creada exitosamente"
    except Exception as e:
        error_message = f"Error al crear estructura de directorios: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Registrar error
        add_to_execution_path(
            "setup_directories_error",
            error_message
        )
        
        return f"Error: {error_message}"

def check_reference_excel_exists():
    """
    Verifica si existe el archivo Excel de referencias.
    
    Returns:
        Mensaje con el estado y ruta del archivo si existe
    """
    logger.info("Verificando existencia del archivo Excel de referencias")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "check_references",
        "Verificando archivo Excel de referencias"
    )
    
    try:
        # Directorio de referencias
        reference_dir = Path("documentos/Referencias")
        
        # Verificar si el directorio existe
        if not reference_dir.exists():
            return json.dumps({
                "status": "error",
                "message": f"Directorio de referencias no encontrado: {reference_dir}",
                "file_exists": False,
                "file_path": None
            })
        
        # Buscar archivos Excel
        excel_files = list(reference_dir.glob("*.xlsx"))
        if not excel_files:
            excel_files = list(reference_dir.glob("*.xls"))
        
        if not excel_files:
            return json.dumps({
                "status": "warning",
                "message": f"No se encontraron archivos Excel en {reference_dir}",
                "file_exists": False,
                "file_path": None
            })
        
        # Obtener el primer archivo Excel
        excel_path = str(excel_files[0])
        
        return json.dumps({
            "status": "success",
            "message": f"Archivo Excel de referencias encontrado: {excel_path}",
            "file_exists": True,
            "file_path": excel_path
        })
    except Exception as e:
        error_message = f"Error al verificar archivo Excel de referencias: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        return json.dumps({ 
            "status": "error",
            "message": error_message,
            "file_exists": False,
            "file_path": None
        })