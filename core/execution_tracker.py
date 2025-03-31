#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seguimiento de la ejecución del agente multiagente.
"""

import logging

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

# Registro de la ruta tomada durante la ejecución
execution_path = []

def reset_execution_path():
    """Reinicia el registro de ejecución"""
    global execution_path
    execution_path = []
    logger.info("Historial de ejecución reiniciado")

def add_to_execution_path(node_name, description):
    """
    Añade un paso al historial de ejecución.
    
    Args:
        node_name: Nombre del nodo o paso
        description: Descripción de la acción realizada
    """
    global execution_path
    execution_path.append({
        "node": node_name,
        "description": description
    })
    logger.debug(f"Registro añadido a historial: {node_name} - {description}")

def get_execution_path():
    """
    Devuelve la copia del historial de ejecución actual.
    
    Returns:
        Lista con el historial de ejecución
    """
    return execution_path.copy()
