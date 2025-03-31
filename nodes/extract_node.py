#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para extraer texto de archivos PDF.
"""

import logging
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.pdf_tools import extract_text_from_pdf

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def extract_text_node(state: TDRAgentState) -> TDRAgentState:
    """
    Extrae texto del PDF del TDR.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con el texto extraído
    """
    logger.info(f"Iniciando extract_text_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "extract_text_node",
        "Nodo de extracción de texto del PDF"
    )
    
    # Verificar que exista la ruta al PDF
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        logger.error("No se proporcionó la ruta del PDF")
        state["messages"].append(AIMessage(content="Error: No se proporcionó la ruta del PDF"))
        state["next_step"] = "end"
        return state
    
    # Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    if text.startswith("Error:"):
        logger.error(f"Error al extraer texto: {text}")
        state["messages"].append(AIMessage(content=text))
        state["next_step"] = "end"
        return state
    
    # Guardar el texto extraído en el estado
    state["tdr_text"] = text
    logger.info(f"Texto extraído: {len(text)} caracteres")
    
    # Actualizar el mensaje en el historial
    state["messages"].append(AIMessage(content=f"Texto extraído exitosamente: {len(text)} caracteres"))
    
    # Siguiente paso
    state["next_step"] = "analyze_tdr"
    
    logger.info(f"Estado después de extract_text_node: {format_state_for_log(state)}")
    return state
