#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para combinar todas las secciones en una propuesta completa.
"""

import json
import logging
import re
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.generation_tools import combine_sections
from tools.evaluation_tools import save_proposal_to_txt

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def combine_proposal_node(state: TDRAgentState) -> TDRAgentState:
    """
    Combina todas las secciones en una propuesta completa.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la propuesta final
    """
    logger.info(f"Iniciando combine_proposal_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "combine_proposal_node",
        "Nodo de integración de la propuesta"
    )
    
    # Verificar que existan secciones generadas
    sections = state.get("sections", [])
    if not sections:
        logger.error("No hay secciones para combinar")
        state["messages"].append(AIMessage(content="Error: No hay secciones para combinar en una propuesta"))
        state["next_step"] = "end"
        return state
    
    # Combinar las secciones con formato JSON
    params = json.dumps({"sections": sections})
    proposal = combine_sections(params)
    if proposal.startswith("Error:"):
        logger.error(f"Error al combinar propuesta: {proposal}")
        state["messages"].append(AIMessage(content=proposal))
        state["next_step"] = "end"
        return state
    
    # Guardar la propuesta en el estado
    state["proposal"] = proposal
    logger.info(f"Propuesta combinada: {len(proposal)} caracteres")
    
    # Extraer nombre del TDR para usarlo en el nombre del archivo
    tdr_name = "TDR"
    try:
        tdr_info = state.get("tdr_info", "")
        # Intentar extraer el título del proyecto del TDR si está en formato JSON
        try:
            tdr_info_dict = json.loads(tdr_info)
            tdr_name = tdr_info_dict.get("titulo_proyecto", "TDR")
        except:
            # Si no está en formato JSON, intentar extraer con regex
            title_match = re.search(r"título del proyecto[:\s]+(.*?)(?:\n|$)", tdr_info, re.IGNORECASE)
            if title_match:
                tdr_name = title_match.group(1).strip()
    except Exception as e:
        logger.warning(f"No se pudo extraer el nombre del TDR: {str(e)}")
    
    # Guardar la propuesta en un archivo
    params = json.dumps({
        "proposal": proposal,
        "tdr_name": tdr_name
    })
    filename = save_proposal_to_txt(params)
    
    if filename.startswith("Error:"):
        logger.warning(f"Error al guardar propuesta: {filename}")
        # Continuamos a pesar del error de guardado
    else:
        logger.info(f"Propuesta guardada en: {filename}")
        state["proposal_filename"] = filename
    
    # Actualizar mensaje en el historial
    state["messages"].append(AIMessage(content=f"Propuesta técnica generada y combinada exitosamente con {len(sections)} secciones"))
    
    # Siguiente paso
    state["next_step"] = "evaluate_proposal"
    
    logger.info(f"Estado después de combine_proposal_node: {format_state_for_log(state)}")
    return state