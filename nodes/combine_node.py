#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para combinar todas las secciones en una propuesta completa.
"""

import json
import logging
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.generation_tools import combine_sections
from tools.evaluation_tools import save_proposal_to_txt

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def combine_proposal_node(state: TDRAgentState) -> TDRAgentState:
    """
    Combina todas las secciones en una propuesta completa con formato mejorado y coherente.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la propuesta final
    """
    logger.info(f"Iniciando combine_proposal_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "combine_proposal_node",
        "Nodo de integración y coherencia de la propuesta"
    )
    
    # Verificar que existan secciones generadas
    sections = state.get("sections", [])
    if not sections:
        logger.error("No hay secciones para combinar")
        state["messages"].append(AIMessage(content="Error: No hay secciones para combinar en una propuesta"))
        state["next_step"] = "end"
        return state
    
    # Extraer metadatos del proyecto
    project_metadata = extract_project_metadata(state)
    
    # Combinar las secciones con formato coherente
    proposal = create_formatted_proposal(sections, project_metadata)
    
    # Aplicar mejoras finales de coherencia y calidad
    improved_proposal = improve_proposal_coherence(proposal)
    
    # Guardar la propuesta en el estado
    state["proposal"] = improved_proposal
    logger.info(f"Propuesta combinada y mejorada: {len(improved_proposal)} caracteres")
    
    # Guardar la propuesta en un archivo
    try:
        # Preparar parámetros para el guardado
        params = json.dumps({
            "proposal": improved_proposal,
            "tdr_name": project_metadata.get("titulo_proyecto", "TDR"),
            "filename": f"Propuesta_{project_metadata.get('titulo_proyecto', 'TDR')}_{get_timestamp()}.txt"
        })
        
        # Guardar el archivo
        filename = save_proposal_to_txt(params)
        
        if filename.startswith("Error:"):
            logger.warning(f"Error al guardar propuesta: {filename}")
        else:
            logger.info(f"Propuesta guardada en: {filename}")
            state["proposal_filename"] = filename
    except Exception as e:
        logger.warning(f"Error al guardar la propuesta: {str(e)}")
    
    # Actualizar mensaje en el historial
    state["messages"].append(AIMessage(
        content=f"Propuesta técnica generada y combinada exitosamente con {len(sections)} secciones. "
                f"Título del proyecto: {project_metadata.get('titulo_proyecto', 'No especificado')}"
    ))
    
    # Siguiente paso
    state["next_step"] = "evaluate_proposal"
    
    logger.info(f"Estado después de combine_proposal_node: {format_state_for_log(state)}")
    return state

def extract_project_metadata(state: TDRAgentState) -> dict:
    """
    Extrae metadatos del proyecto del estado actual.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Diccionario con metadatos del proyecto
    """
    metadata = {
        "titulo_proyecto": "Proyecto No Especificado",
        "cliente": "Cliente No Especificado",
        "fecha": get_current_date()
    }
    
    try:
        # Intentar obtener información del TDR
        tdr_info = state.get("tdr_info", "")
        
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
            # Extraer campos relevantes si existen
            if "titulo_proyecto" in tdr_dict and tdr_dict["titulo_proyecto"] != "No especificado":
                metadata["titulo_proyecto"] = tdr_dict["titulo_proyecto"]
            
            if "cliente" in tdr_dict and tdr_dict["cliente"] != "No especificado":
                metadata["cliente"] = tdr_dict["cliente"]
            
            # Buscar fecha en varios campos posibles
            date_fields = ["fecha", "fecha_inicio", "plazos", "cronograma"]
            for field in date_fields:
                if field in tdr_dict and tdr_dict[field] and tdr_dict[field] != "No especificado":
                    metadata["fecha"] = tdr_dict[field]
                    break
        else:
            # Si no es JSON, buscar en el texto del TDR o en las secciones generadas
            # Búsqueda en la primera sección (introducción) si existe
            if len(state.get("sections", [])) > 0:
                intro_section = state["sections"][0]
                
                # Buscar título del proyecto
                title_match = re.search(r"(?:título|titulo|nombre)\s+del\s+proyecto[:\s]+(.+?)(?=\n|$)", 
                                       intro_section, re.IGNORECASE)
                if title_match:
                    metadata["titulo_proyecto"] = title_match.group(1).strip()
                
                # Buscar cliente
                client_match = re.search(r"(?:cliente|empresa)[:\s]+(.+?)(?=\n|$)", 
                                        intro_section, re.IGNORECASE)
                if client_match:
                    metadata["cliente"] = client_match.group(1).strip()
    except Exception as e:
        logger.warning(f"Error al extraer metadatos: {str(e)}")
    
    return metadata

def create_formatted_proposal(sections: list, metadata: dict) -> str:
    """
    Crea una propuesta formateada a partir de las secciones y metadatos.
    
    Args:
        sections: Lista de secciones generadas
        metadata: Metadatos del proyecto
        
    Returns:
        Propuesta técnica formateada
    """
    # Crear encabezado y título principal
    titulo_proyecto = metadata.get("titulo_proyecto", "Proyecto No Especificado")
    cliente = metadata.get("cliente", "Cliente No Especificado")
    fecha = metadata.get("fecha", get_current_date())
    
    proposal = f"""# PROPUESTA TÉCNICA
**Documento: PKS-537 RQ-01**
**Fecha: {fecha}**
**Proyecto: {titulo_proyecto}**
**Cliente: {cliente}**

---

"""
    
    # Añadir tabla de contenido
    proposal += "## Tabla de Contenido\n\n"
    section_pattern = r"##\s+(\d+)\.\s+([^\n]+)"
    toc_items = []
    
    for i, section in enumerate(sections):
        section_matches = re.findall(section_pattern, section)
        if section_matches:
            for num, title in section_matches:
                toc_items.append(f"{num}. {title}")
        else:
            # Si no se encuentra el patrón, usar el índice como número
            section_title = section.split("\n")[0].replace("#", "").strip()
            toc_items.append(f"{i+1}. {section_title}")
    
    proposal += "\n".join(toc_items) + "\n\n---\n\n"
    
    # Combinar secciones asegurando formato coherente
    cleaned_sections = []
    for i, section in enumerate(sections):
        # Asegurar que cada sección tenga un número correcto
        section_number_pattern = r"##\s+\d+\."
        if not re.search(section_number_pattern, section):
            # Si no tiene número, añadirlo
            section_lines = section.split("\n")
            if section_lines and section_lines[0].startswith("##"):
                section_lines[0] = f"## {i+1}. {section_lines[0].replace('#', '').strip()}"
                section = "\n".join(section_lines)
        
        # Normalizar subsecciones
        section_number = i + 1
        subsection_pattern = r"###\s+([^0-9\n].*)"
        section = re.sub(
            subsection_pattern,
            f"### {section_number}.\\1",
            section
        )
        
        cleaned_sections.append(section)
    
    proposal += "\n\n".join(cleaned_sections)
    
    # Añadir sección final y número de versión
    proposal += "\n\n---\n\n**Versión 1.0**\n\n"
    proposal += "**Documento preparado en conformidad con los requisitos del documento PKS-537 RQ-01**"
    
    return proposal

def improve_proposal_coherence(proposal: str) -> str:
    """
    Mejora la coherencia global de la propuesta.
    
    Args:
        proposal: Texto de la propuesta completa
        
    Returns:
        Propuesta con coherencia mejorada
    """
    # Limpiar etiquetas think
    proposal = re.sub(r'<think>.*?</think>', '', proposal, flags=re.DOTALL)
    
    # Eliminar caracteres no latinos
    proposal = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', proposal)
    
    # Normalizar formato de subsecciones
    section_pattern = r'##\s+(\d+)\.\s+'
    sections = re.findall(section_pattern, proposal)
    
    for section_num in sections:
        # Normalizar subsecciones para esta sección
        subsection_pattern = r'###\s+[^\d]'
        subsection_replacement = f'### {section_num}.'
        proposal = re.sub(
            subsection_pattern,
            subsection_replacement,
            proposal
        )
    
    # Eliminar líneas vacías múltiples
    proposal = re.sub(r'\n\s*\n\s*\n+', '\n\n', proposal)
    
    # Corregir formato de listas para mayor consistencia
    proposal = re.sub(r'(?<=\n)\*\s+', '- ', proposal)
    proposal = re.sub(r'(?<=\n)•\s+', '- ', proposal)
    
    return proposal

def get_current_date() -> str:
    """
    Obtiene la fecha actual en formato legible.
    
    Returns:
        Fecha actual formateada
    """
    return datetime.now().strftime("%d de %B, %Y")

def get_timestamp() -> str:
    """
    Obtiene un timestamp para nombrar archivos.
    
    Returns:
        Timestamp en formato YYYYMMDDHHMMSS
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")