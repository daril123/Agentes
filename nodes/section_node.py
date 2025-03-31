#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para generar las secciones de la propuesta técnica.
"""

import json
import logging
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.generation_tools import generate_section

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def generate_sections_node(state: TDRAgentState) -> TDRAgentState:
    """
    Genera una sección de la propuesta técnica a la vez.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la nueva sección generada
    """
    logger.info(f"Iniciando generate_sections_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "generate_sections_node",
        "Nodo de generación de secciones"
    )
    
    # Verificar que existan el índice y la información del TDR
    index = state.get("index")
    tdr_info = state.get("tdr_info")
    current_index = state.get("current_section_index", 0)
    
    if not index or not tdr_info:
        logger.error("Faltan datos necesarios para generar secciones")
        state["messages"].append(AIMessage(content="Error: Faltan datos necesarios para generar secciones"))
        state["next_step"] = "end"
        return state
    
    # Inicializar el registro de contenido generado si no existe
    if "generated_content" not in state:
        state["generated_content"] = {}
    
    # Obtener lista de secciones
    section_names = list(index.keys())
    
    # Verificar si ya se procesaron todas las secciones
    if current_index >= len(section_names):
        logger.info("Todas las secciones han sido generadas")
        state["messages"].append(AIMessage(content="Todas las secciones han sido generadas exitosamente"))
        state["next_step"] = "combine_proposal"
        return state
    
    # Obtener la sección actual
    section_name = section_names[current_index]
    section_description = index[section_name]
    
    logger.info(f"Generando sección {current_index+1}/{len(section_names)}: {section_name}")
    
    # Obtener el contenido ya generado para pasarlo como contexto
    previously_generated = ""
    for prev_section, content in state.get("generated_content", {}).items():
        previously_generated += f"## {prev_section}\n\n{content}\n\n"
    
    # Generar la sección - pasar parámetros como un solo JSON con el contenido previo
    params = json.dumps({
        "section_name": section_name,
        "description": section_description,
        "info": tdr_info,
        "previous_content": previously_generated  # Añadir contenido previo
    })
    
    # Llamar a la herramienta de generación con contexto de propuestas similares
    section_content = generate_section(params)
    
    if section_content.startswith("Error:"):
        logger.error(f"Error al generar sección {section_name}: {section_content}")
        state["messages"].append(AIMessage(content=section_content))
        
        # Crear una sección mínima para que el flujo pueda continuar
        minimal_section = f"## {section_name}\n\n[Esta sección no pudo generarse correctamente. Por favor, revise los logs para más detalles.]"
        sections = state.get("sections", [])
        sections.append(minimal_section)
        state["sections"] = sections
        
        logger.info(f"Se añadió una sección mínima para {section_name} debido a un error")
    else:
        # Añadir la sección generada a la lista
        sections = state.get("sections", [])
        sections.append(section_content)
        state["sections"] = sections
        
        # Guardar el contenido generado en el registro para contexto futuro
        # Extraer el contenido sin el encabezado para almacenarlo
        content_without_header = section_content
        if section_content.startswith(f"## {section_name}"):
            content_without_header = section_content[len(f"## {section_name}"):]
        state["generated_content"][section_name] = content_without_header.strip()
        
        logger.info(f"Sección {section_name} generada: {len(section_content)} caracteres")
    
    # Actualizar mensaje en el historial
    state["messages"].append(AIMessage(content=f"Sección {current_index+1}/{len(section_names)} ({section_name}) generada"))
    
    # Incrementar el índice para la próxima sección
    state["current_section_index"] = current_index + 1
    
    # Mantener el mismo paso para generar la siguiente sección
    state["next_step"] = "generate_sections"
    
    logger.info(f"Estado después de generate_sections_node: {format_state_for_log(state)}")
    return state
