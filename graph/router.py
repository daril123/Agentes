#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router para decidir el siguiente paso en el flujo de trabajo.
"""

import logging
from typing import Literal
from core.state import TDRAgentState
from core.execution_tracker import add_to_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def router(state: TDRAgentState) -> Literal["extract_text", "analyze_tdr", "generate_index", "generate_sections", "combine_proposal", "evaluate_proposal", "end"]:
    """
    Determina el siguiente paso en el flujo de trabajo.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Próximo nodo a ejecutar en el flujo
    """
    next_step = state["next_step"]
    logger.info(f"Router decidiendo siguiente paso: {next_step}")
    
    # Registrar decisión del router
    add_to_execution_path(
        "router",
        f"Siguiente paso: {next_step}"
    )
    
    return next_step
