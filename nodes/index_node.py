#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para generar el índice de la propuesta técnica.
"""

import json
import logging
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.analysis_tools import generate_index
import re
# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def generate_index_node(state: TDRAgentState) -> TDRAgentState:
    """
    Genera el índice para la propuesta técnica.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con el índice de la propuesta
    """
    logger.info(f"Iniciando generate_index_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "generate_index_node",
        "Nodo de generación del índice"
    )
    
    # Verificar que exista la información del TDR
    tdr_info = state.get("tdr_info")
    if not tdr_info:
        logger.error("No hay información del TDR para generar el índice")
        state["messages"].append(AIMessage(content="Error: No hay información del TDR para generar el índice"))
        state["next_step"] = "end"
        return state
    
    # Limpiamos tags <think> y </think> de la información del TDR
    # Esto puede ayudar si el problema está también en la entrada
    tdr_info_clean = re.sub(r'<think>.*?</think>', '', tdr_info, flags=re.DOTALL)
    tdr_info_clean = re.sub(r'<think>.*', '', tdr_info_clean, flags=re.DOTALL)  # Por si no hay tag de cierre
    
    if tdr_info_clean.strip() == "":
        # Si al limpiar quedó vacío, usamos el original
        tdr_info_clean = tdr_info
    
    # Generar el índice con la información limpia
    index_str = generate_index(tdr_info_clean)
    
    try:
        # Intentar convertir el índice a diccionario
        index_dict = json.loads(index_str)
        
        # Guardar el índice en el estado
        state["index"] = index_dict
        state["sections"] = []  # Inicializar lista de secciones
        state["current_section_index"] = 0  # Inicializar índice para generar secciones
        
        # Crear una lista con los nombres de las secciones
        section_names = list(index_dict.keys())
        
        logger.info(f"Índice generado con {len(section_names)} secciones")
        
        # Actualizar el mensaje en el historial
        index_summary = ", ".join(section_names[:5])
        if len(section_names) > 5:
            index_summary += f", y {len(section_names) - 5} más"
        
        state["messages"].append(AIMessage(content=f"Índice generado exitosamente con las siguientes secciones: {index_summary}"))
        
        # Siguiente paso
        state["next_step"] = "generate_sections"
        
    except json.JSONDecodeError as e:
        error_message = f"Error al decodificar el índice JSON: {str(e)}, índice: {index_str[:200]}..."
        logger.error(error_message)
        
        # Intentar reparar el JSON antes de fallar
        try:
            # Limpieza avanzada del JSON
            # Eliminar tags <think> y </think>
            fixed_index_str = re.sub(r'<think>.*?</think>', '', index_str, flags=re.DOTALL)
            fixed_index_str = re.sub(r'<think>.*', '', fixed_index_str, flags=re.DOTALL)  # Por si no hay tag de cierre
            
            # Eliminar cualquier texto antes de la primera llave de apertura
            fixed_index_str = re.sub(r'^[^{]*', '', fixed_index_str)
            
            # Eliminar cualquier texto después de la última llave de cierre
            fixed_index_str = re.sub(r'[^}]*$', '', fixed_index_str)
            
            # Limpieza básica de JSON
            fixed_index_str = re.sub(r',\s*}', '}', fixed_index_str)
            fixed_index_str = re.sub(r',\s*]', ']', fixed_index_str)
            
            # Intentar cargar el JSON reparado
            try:
                index_dict = json.loads(fixed_index_str)
                
                # Si llegamos aquí, la reparación funcionó
                logger.info("JSON reparado exitosamente")
                state["index"] = index_dict
                state["sections"] = []
                state["current_section_index"] = 0
                
                section_names = list(index_dict.keys())
                index_summary = ", ".join(section_names[:5])
                if len(section_names) > 5:
                    index_summary += f", y {len(section_names) - 5} más"
                
                state["messages"].append(AIMessage(content=f"Índice reparado y generado con las siguientes secciones: {index_summary}"))
                
                # Siguiente paso
                state["next_step"] = "generate_sections"
                
            except json.JSONDecodeError:
                # Si aún falla, crear un índice predeterminado
                logger.warning("Reparación del JSON fallida. Creando índice predeterminado.")
                
                default_index = {
                    "introduccion": "Introducción y contexto del proyecto",
                    "objetivos": "Objetivos generales y específicos del proyecto",
                    "alcance_trabajo": "Alcance detallado del trabajo a realizar",
                    "metodologia": "Metodología propuesta para el desarrollo del proyecto",
                    "plan_trabajo_cronograma": "Plan de trabajo y cronograma de actividades",
                    "entregables": "Entregables con descripción detallada",
                    "recursos_humanos_tecnicos": "Recursos humanos y técnicos asignados",
                    "gestion_riesgos": "Gestión de riesgos del proyecto",
                    "plan_calidad": "Plan de calidad del proyecto"
                }
                
                state["index"] = default_index
                state["sections"] = []
                state["current_section_index"] = 0
                
                section_names = list(default_index.keys())
                index_summary = ", ".join(section_names)
                
                state["messages"].append(AIMessage(content=f"Se ha creado un índice predeterminado con las siguientes secciones: {index_summary}"))
                
                # Siguiente paso
                state["next_step"] = "generate_sections"
        except Exception as e2:
            # Si todo falla, crear un índice predeterminado
            logger.error(f"Error al intentar reparar el JSON: {str(e2)}")
            
            default_index = {
                "introduccion": "Introducción y contexto del proyecto",
                "objetivos": "Objetivos generales y específicos del proyecto",
                "alcance_trabajo": "Alcance detallado del trabajo a realizar",
                "metodologia": "Metodología propuesta para el desarrollo del proyecto",
                "plan_trabajo_cronograma": "Plan de trabajo y cronograma de actividades",
                "entregables": "Entregables con descripción detallada",
                "recursos_humanos_tecnicos": "Recursos humanos y técnicos asignados",
                "gestion_riesgos": "Gestión de riesgos del proyecto",
                "plan_calidad": "Plan de calidad del proyecto"
            }
            
            state["index"] = default_index
            state["sections"] = []
            state["current_section_index"] = 0
            
            section_names = list(default_index.keys())
            index_summary = ", ".join(section_names)
            
            state["messages"].append(AIMessage(content=f"Se ha creado un índice predeterminado con las siguientes secciones: {index_summary}"))
            
            # Siguiente paso
            state["next_step"] = "generate_sections"
    
    logger.info(f"Estado después de generate_index_node: {format_state_for_log(state)}")
    return state