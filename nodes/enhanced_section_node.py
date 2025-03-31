#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo mejorado para generar las secciones de la propuesta técnica con mayor coherencia.
"""

import json
import logging
import re
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.section_generator import generate_coherent_section

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def enhanced_generate_sections_node(state: TDRAgentState) -> TDRAgentState:
    """
    Genera una sección de la propuesta técnica con formato coherente y estandarizado.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la nueva sección generada
    """
    logger.info(f"Iniciando enhanced_generate_sections_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "enhanced_generate_sections_node",
        "Nodo mejorado de generación de secciones"
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
    
    # Extraer metadatos del proyecto desde el TDR
    project_metadata = extract_project_metadata(tdr_info)
    
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
    
    # Obtener las secciones anteriores para contexto
    previous_sections = []
    for prev_name in section_names[:current_index]:
        if prev_name in state.get("generated_content", {}):
            previous_sections.append(state["generated_content"][prev_name])
    
    # Preparar parámetros para la generación coherente
    params = json.dumps({
        "section_name": section_name,
        "section_number": current_index + 1,
        "description": section_description,
        "tdr_info": tdr_info,
        "previous_sections": previous_sections,
        "proposal_metadata": project_metadata
    })
    
    # Generar la sección con el formato mejorado
    section_content = generate_coherent_section(params)
    
    if "Error" in section_content:
        logger.error(f"Error al generar sección {section_name}: {section_content}")
        state["messages"].append(AIMessage(content=f"Error en la generación de la sección {section_name}"))
        
        # Crear una sección mínima para que el flujo pueda continuar
        minimal_section = f"## {current_index+1}. {section_name.upper()}\n\n[Esta sección no pudo generarse correctamente. Por favor, revise los logs para más detalles.]"
        sections = state.get("sections", [])
        sections.append(minimal_section)
        state["sections"] = sections
        
        logger.info(f"Se añadió una sección mínima para {section_name} debido a un error")
    else:
        # Limpiar la sección de cualquier contenido no deseado
        clean_content = clean_section(section_content)
        
        # Añadir la sección generada a la lista
        sections = state.get("sections", [])
        sections.append(clean_content)
        state["sections"] = sections
        
        # Guardar el contenido generado en el registro para contexto futuro
        state["generated_content"][section_name] = clean_content
        
        logger.info(f"Sección {section_name} generada: {len(clean_content)} caracteres")
    
    # Actualizar mensaje en el historial
    state["messages"].append(AIMessage(content=f"Sección {current_index+1}/{len(section_names)} ({section_name}) generada"))
    
    # Incrementar el índice para la próxima sección
    state["current_section_index"] = current_index + 1
    
    # Mantener el mismo paso para generar la siguiente sección
    state["next_step"] = "generate_sections"
    
    logger.info(f"Estado después de enhanced_generate_sections_node: {format_state_for_log(state)}")
    return state

def extract_project_metadata(tdr_info: str) -> dict:
    """
    Extrae metadatos del proyecto desde la información del TDR.
    
    Args:
        tdr_info: Información extraída del TDR
        
    Returns:
        Diccionario con metadatos del proyecto
    """
    metadata = {
        "titulo_proyecto": "Proyecto No Especificado",
        "cliente": "Cliente No Especificado",
        "fecha": "No Especificada"
    }
    
    try:
        # Intentar parsear como JSON primero
        info_dict = json.loads(tdr_info)
        
        # Extraer campos relevantes si existen
        if "titulo_proyecto" in info_dict:
            metadata["titulo_proyecto"] = info_dict["titulo_proyecto"]
        
        if "cliente" in info_dict:
            metadata["cliente"] = info_dict["cliente"]
        
        # Intentar extraer fecha si existe algún campo relacionado
        date_fields = ["fecha", "fecha_inicio", "fecha_elaboracion", "fecha_entrega"]
        for field in date_fields:
            if field in info_dict:
                metadata["fecha"] = info_dict[field]
                break
    except json.JSONDecodeError:
        # Si no es JSON, intentar extraer con regex
        title_match = re.search(r"(?:título|titulo|nombre)\s+del\s+proyecto[:\s]+(.+?)(?:\n|$)", tdr_info, re.IGNORECASE)
        if title_match:
            metadata["titulo_proyecto"] = title_match.group(1).strip()
        
        client_match = re.search(r"(?:cliente|empresa)[:\s]+(.+?)(?:\n|$)", tdr_info, re.IGNORECASE)
        if client_match:
            metadata["cliente"] = client_match.group(1).strip()
        
        date_match = re.search(r"(?:fecha)[:\s]+(.+?)(?:\n|$)", tdr_info, re.IGNORECASE)
        if date_match:
            metadata["fecha"] = date_match.group(1).strip()
    
    return metadata

def clean_section(section_content: str) -> str:
    """
    Limpia una sección generada para eliminar elementos no deseados.
    
    Args:
        section_content: Contenido a limpiar
        
    Returns:
        Contenido limpio
    """
    # Eliminar etiquetas think
    content = re.sub(r'<think>.*?</think>', '', section_content, flags=re.DOTALL)
    
    # Eliminar caracteres no latinos (excepto puntuación común)
    content = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', content)
    
    # Eliminar líneas vacías múltiples
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Normalizar encabezados (###, ####)
    # Asegurar que los encabezados de nivel 2 (##) tengan el formato correcto
    content = re.sub(r'^##(?!\s+\d+\.)', r'## 1.', content, flags=re.MULTILINE)
    
    return content.strip()