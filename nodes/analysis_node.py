#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para analizar el TDR y extraer información clave.
"""

import logging
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.analysis_tools import analyze_tdr

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def analyze_tdr_node(state: TDRAgentState) -> TDRAgentState:
    """
    Analiza el TDR para extraer información clave.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la información extraída del TDR
    """
    logger.info(f"Iniciando analyze_tdr_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "analyze_tdr_node",
        "Nodo de análisis del TDR"
    )
    
    # Verificar que exista el texto del TDR
    tdr_text = state.get("tdr_text")
    if not tdr_text:
        logger.error("No hay texto del TDR para analizar")
        state["messages"].append(AIMessage(content="Error: No hay texto del TDR para analizar"))
        state["next_step"] = "end"
        return state
    
    # Analizar el TDR
    tdr_info = analyze_tdr(tdr_text)
    if tdr_info.startswith("Error:"):
        logger.error(f"Error al analizar TDR: {tdr_info}")
        state["messages"].append(AIMessage(content=tdr_info))
        state["next_step"] = "end"
        return state
    
    # Guardar la información extraída en el estado
    state["tdr_info"] = tdr_info
    logger.info(f"TDR analizado: {len(tdr_info)} caracteres")
    
    # Actualizar el mensaje en el historial
    state["messages"].append(AIMessage(content=f"TDR analizado exitosamente. Información extraída: {tdr_info[:200]}..."))
    
    # Siguiente paso
    state["next_step"] = "generate_index"
    
    logger.info(f"Estado después de analyze_tdr_node: {format_state_for_log(state)}")
    return state
