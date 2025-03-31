#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definición del estado para el sistema multiagente de análisis de TDR.
"""

import logging
from typing import TypedDict, Annotated, List, Literal, Union, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

# Modificación en core/state.py
class TDRAgentState(TypedDict):
    """
    Define el estado para el sistema multiagente de análisis de TDR.
    Almacena la información de estado a medida que fluye por el grafo.
    """
    messages: List[Annotated[Union[SystemMessage, HumanMessage, AIMessage, FunctionMessage], "Historial de la conversación"]]
    next_step: Literal["extract_text", "analyze_tdr", "generate_index", "generate_sections", "combine_proposal", "evaluate_proposal", "end"]
    tdr_text: Optional[str]  # Texto extraído del PDF
    tdr_info: Optional[str]  # Información extraída del TDR
    index: Optional[Dict[str, str]]  # Índice y descripciones
    sections: Optional[List[str]]  # Secciones generadas
    proposal: Optional[str]  # Propuesta final
    evaluation: Optional[str]  # Evaluación de la propuesta
    pdf_path: Optional[str]  # Ruta al archivo PDF
    current_section_index: Optional[int]  # Índice de la sección actual que se está generando
    generated_content: Optional[Dict[str, str]]  # Registro de contenido ya generado por sección
    
def format_state_for_log(state): 
    """
    Formatea el estado del agente para mostrar en logs.
    
    Args:
        state: Estado del agente a formatear
        
    Returns:
        Diccionario con el estado formateado para logs
    """
    messages = []
    for msg in state.get("messages", []):
        msg_type = type(msg).__name__
        if isinstance(msg, SystemMessage):
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            messages.append(f"[{msg_type}]: {content}")
        else:
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            messages.append(f"[{msg_type}]: {content}")
    
    formatted = {
        "message_count": len(messages),
        "message_types": messages,
        "next_step": state.get("next_step", None)
    }
    
    # Añadir otros campos si existen
    for key in state:
        if key not in ["messages", "next_step"]:
            if key in ["tdr_text", "tdr_info", "proposal", "evaluation"] and state.get(key):
                # Truncar campos de texto largos
                formatted[key] = state[key][:100] + "..." if state[key] and len(state[key]) > 100 else state[key]
            else:
                formatted[key] = state[key]
            
    return formatted
