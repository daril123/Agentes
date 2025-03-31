#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construcción del grafo del agente multiagente para análisis de TDR.
"""

import logging
from langgraph.graph import StateGraph, END
from core.state import TDRAgentState
from graph.router import router
from nodes.extract_node import extract_text_node
from nodes.analysis_node import analyze_tdr_node
from nodes.index_node import generate_index_node
from nodes.section_node import generate_sections_node
from nodes.combine_node import combine_proposal_node
from nodes.evaluate_node import evaluate_proposal_node

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def build_tdr_agent():
    """
    Construye y compila el grafo del agente para análisis de TDR.
    
    Returns:
        Grafo compilado listo para ser ejecutado
    """
    logger.info("Construyendo el grafo del agente para análisis de TDR")
    
    # Inicializar el grafo con el tipo de estado
    workflow = StateGraph(TDRAgentState)
    
    # Agregar nodos
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("analyze_tdr", analyze_tdr_node)
    workflow.add_node("generate_index", generate_index_node)
    workflow.add_node("generate_sections", generate_sections_node)
    workflow.add_node("combine_proposal", combine_proposal_node)
    workflow.add_node("evaluate_proposal", evaluate_proposal_node)
    logger.info("Nodos añadidos al grafo")
    
    # Establecer el nodo de inicio
    workflow.set_entry_point("extract_text")
    logger.info("Punto de entrada establecido: extract_text")
    
    # Conectar los nodos según la decisión del router
    workflow.add_conditional_edges("extract_text", router, {
        "analyze_tdr": "analyze_tdr",
        "end": END
    })
    
    workflow.add_conditional_edges("analyze_tdr", router, {
        "generate_index": "generate_index",
        "end": END
    })
    
    workflow.add_conditional_edges("generate_index", router, {
        "generate_sections": "generate_sections",
        "end": END
    })
    
    workflow.add_conditional_edges("generate_sections", router, {
        "generate_sections": "generate_sections",  # Loop para generar todas las secciones
        "combine_proposal": "combine_proposal",
        "end": END
    })
    
    workflow.add_conditional_edges("combine_proposal", router, {
        "evaluate_proposal": "evaluate_proposal",
        "end": END
    })
    
    workflow.add_conditional_edges("evaluate_proposal", router, {
        "end": END
    })
    
    logger.info("Bordes condicionales añadidos")
    
    # Compilar el grafo
    logger.info("Compilando el grafo")
    return workflow.compile()
