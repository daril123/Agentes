#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para generar las secciones de la propuesta técnica.
"""

import json
import logging
import re
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.generation_tools import generate_section

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def generate_sections_node(state: TDRAgentState) -> TDRAgentState:
    """
    Genera una sección de la propuesta técnica a la vez con mejoras de especificidad y formato.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la nueva sección generada
    """
    logger.info(f"Iniciando generate_sections_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "generate_sections_node",
        "Nodo de generación de secciones específicas"
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
        if len(previously_generated) < 3000:  # Limitar el contexto a 3000 caracteres
            previously_generated += f"## {prev_section}\n\n{content}\n\n"
    
    # Extraer metadatos del proyecto a partir del TDR
    project_metadata = extract_project_metadata(tdr_info)
    
    # Generar la sección - pasar parámetros como un solo JSON con todos los datos relevantes
    params = json.dumps({
        "section_name": section_name,
        "description": section_description,
        "info": tdr_info,
        "previous_content": previously_generated,
        "project_metadata": project_metadata,
        "section_number": current_index + 1,
        "total_sections": len(section_names)
    })
    
    # Llamar a la herramienta de generación
    section_content = generate_section(params)
    
    if section_content.startswith("Error:"):
        logger.error(f"Error al generar sección {section_name}: {section_content}")
        state["messages"].append(AIMessage(content=section_content))
        
        # Crear una sección mínima para que el flujo pueda continuar
        minimal_section = f"## {current_index+1}. {section_name.upper()}\n\n[Esta sección no pudo generarse correctamente. Por favor, revise los logs para más detalles.]"
        sections = state.get("sections", [])
        sections.append(minimal_section)
        state["sections"] = sections
        
        logger.info(f"Se añadió una sección mínima para {section_name} debido a un error")
    else:
        # Limpiar y mejorar la sección generada
        improved_section = improve_section_content(section_content, section_name, current_index+1)
        
        # Añadir la sección generada a la lista
        sections = state.get("sections", [])
        sections.append(improved_section)
        state["sections"] = sections
        
        # Guardar el contenido generado en el registro para contexto futuro
        # Extraer el contenido sin el encabezado para almacenarlo
        content_without_header = improved_section
        header_pattern = r'^##\s+\d+\.\s+' + re.escape(section_name.upper())
        content_without_header = re.sub(header_pattern, '', content_without_header, flags=re.IGNORECASE)
        
        state["generated_content"][section_name] = content_without_header.strip()
        
        logger.info(f"Sección {section_name} generada y mejorada: {len(improved_section)} caracteres")
    
    # Actualizar mensaje en el historial
    state["messages"].append(AIMessage(content=f"Sección {current_index+1}/{len(section_names)} ({section_name}) generada"))
    
    # Incrementar el índice para la próxima sección
    state["current_section_index"] = current_index + 1
    
    # Mantener el mismo paso para generar la siguiente sección
    state["next_step"] = "generate_sections"
    
    logger.info(f"Estado después de generate_sections_node: {format_state_for_log(state)}")
    return state

def extract_project_metadata(tdr_info: str) -> dict:
    """
    Extrae metadatos clave del proyecto a partir del TDR.
    
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
        # Verificar si es un JSON
        is_json = False
        tdr_dict = {}
        try:
            if isinstance(tdr_info, str):
                tdr_dict = json.loads(tdr_info)
                is_json = True
        except json.JSONDecodeError:
            pass
        
        if is_json:
            # Extraer campos relevantes si existen en el JSON
            if "titulo_proyecto" in tdr_dict:
                metadata["titulo_proyecto"] = tdr_dict["titulo_proyecto"]
            
            if "cliente" in tdr_dict:
                metadata["cliente"] = tdr_dict["cliente"]
            
            # Buscar fecha en varios campos posibles
            date_fields = ["fecha", "fecha_inicio", "plazos", "cronograma"]
            for field in date_fields:
                if field in tdr_dict and tdr_dict[field] and tdr_dict[field] != "No especificado":
                    metadata["fecha"] = tdr_dict[field]
                    break
        else:
            # Si no es JSON, intentar extraer con regex
            title_match = re.search(r"(?:título|titulo|nombre)\s+del\s+proyecto[:\s]+(.+?)(?=\n|$)", tdr_info, re.IGNORECASE)
            if title_match:
                metadata["titulo_proyecto"] = title_match.group(1).strip()
            
            client_match = re.search(r"(?:cliente|empresa)[:\s]+(.+?)(?=\n|$)", tdr_info, re.IGNORECASE)
            if client_match:
                metadata["cliente"] = client_match.group(1).strip()
            
            date_match = re.search(r"(?:fecha|plazo)[:\s]+(.+?)(?=\n|$)", tdr_info, re.IGNORECASE)
            if date_match:
                metadata["fecha"] = date_match.group(1).strip()
    except Exception as e:
        logger.warning(f"Error al extraer metadatos del proyecto: {str(e)}")
    
    return metadata

def improve_section_content(content: str, section_name: str, section_number: int) -> str:
    """
    Mejora el contenido de la sección generada para hacerla más específica y mejorar su formato.
    
    Args:
        content: Contenido de la sección generada
        section_name: Nombre de la sección
        section_number: Número de la sección
        
    Returns:
        Contenido mejorado de la sección
    """
    try:
        # Limpiar etiquetas think
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Eliminar caracteres no latinos (excepto puntuación común)
        content = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', content)
        
        # Asegurar que el título de la sección tenga el formato correcto
        formatted_title = f"## {section_number}. {section_name.upper()}"
        
        # Eliminar cualquier encabezado existente que haya generado el LLM
        content_without_header = re.sub(r'^##\s+.*?\n', '', content, flags=re.MULTILINE)
        
        # Normalizar subsecciones (si existen)
        content_with_normalized_subsections = re.sub(
            r'^###\s+(.*?)$', 
            f'### {section_number}.\\1', 
            content_without_header, 
            flags=re.MULTILINE
        )
        
        # Eliminar líneas vacías múltiples
        content_clean = re.sub(r'\n\s*\n\s*\n+', '\n\n', content_with_normalized_subsections)
        
        # Combinar título y contenido
        improved_content = f"{formatted_title}\n\n{content_clean.strip()}"
        
        return improved_content
    except Exception as e:
        logger.warning(f"Error al mejorar el contenido de la sección {section_name}: {str(e)}")
        return content
