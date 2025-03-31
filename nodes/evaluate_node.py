#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para evaluar la calidad y adecuación de la propuesta técnica.
"""

import json
import logging
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.evaluation_tools import evaluate_proposal

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def evaluate_proposal_node(state: TDRAgentState) -> TDRAgentState:
    """
    Evalúa la calidad y adecuación de la propuesta técnica.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la evaluación de la propuesta
    """
    logger.info(f"Iniciando evaluate_proposal_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "evaluate_proposal_node",
        "Nodo de evaluación de la propuesta"
    )
    
    # Verificar que existan la propuesta y la información del TDR
    proposal = state.get("proposal")
    tdr_info = state.get("tdr_info")
    
    if not proposal or not tdr_info:
        logger.error("Faltan datos necesarios para evaluar la propuesta")
        state["messages"].append(AIMessage(content="Error: Faltan datos necesarios para evaluar la propuesta"))
        state["next_step"] = "end"
        return state
    
    # Evaluar la propuesta - pasar parámetros como un solo JSON
    params = json.dumps({
        "proposal": proposal,
        "tdr_info": tdr_info
    })
    evaluation = evaluate_proposal(params)
    if evaluation.startswith("Error:"):
        logger.error(f"Error al evaluar propuesta: {evaluation}")
        state["messages"].append(AIMessage(content=evaluation))
        # Continuamos a pesar del error de evaluación
    else:
        # Guardar la evaluación en el estado
        state["evaluation"] = evaluation
        logger.info(f"Propuesta evaluada: {evaluation[:200]}...")
        
    try:
        # Intentar convertir la evaluación a diccionario para presentarla mejor
        eval_dict = json.loads(evaluation)
        status = eval_dict.get("status", "desconocido")
        puntuacion = eval_dict.get("puntuacion", "N/A")
        
        # Actualizar mensaje en el historial con resumen de la evaluación
        state["messages"].append(AIMessage(
            content=f"Propuesta evaluada: Estado '{status}' con puntuación {puntuacion}/10. "
                   f"La propuesta ha sido completada y guardada en el archivo '{state.get('proposal_filename', 'propuesta.txt')}'."
        ))
    except (json.JSONDecodeError, TypeError):
        # Si hay error al decodificar el JSON, usar texto plano
        state["messages"].append(AIMessage(
            content=f"Evaluación completada. La propuesta ha sido guardada en el archivo '{state.get('proposal_filename', 'propuesta.txt')}'."
        ))
    
    # Siguiente paso (fin del proceso)
    state["next_step"] = "end"
    
    logger.info(f"Estado después de evaluate_proposal_node: {format_state_for_log(state)}")
    return state
