#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo para evaluar la calidad y adecuación de la propuesta técnica.
"""

import json
import logging
import re
from langchain_core.messages import AIMessage
from core.state import TDRAgentState, format_state_for_log
from core.execution_tracker import add_to_execution_path
from tools.evaluation_tools import evaluate_proposal

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def evaluate_proposal_node(state: TDRAgentState) -> TDRAgentState:
    """
    Evalúa la calidad y adecuación de la propuesta técnica con criterios específicos.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        Estado actualizado con la evaluación de la propuesta
    """
    logger.info(f"Iniciando evaluate_proposal_node con estado: {format_state_for_log(state)}")
    
    # Registrar inicio del nodo
    add_to_execution_path(
        "evaluate_proposal_node",
        "Nodo de evaluación detallada de la propuesta"
    )
    
    # Verificar que existan la propuesta y la información del TDR
    proposal = state.get("proposal")
    tdr_info = state.get("tdr_info")
    
    if not proposal or not tdr_info:
        logger.error("Faltan datos necesarios para evaluar la propuesta")
        state["messages"].append(AIMessage(content="Error: Faltan datos necesarios para evaluar la propuesta"))
        state["next_step"] = "end"
        return state
    
    # Extraer requisitos específicos del TDR para la evaluación
    tdr_requirements = extract_tdr_requirements(tdr_info)
    
    # Evaluar la propuesta con requisitos específicos
    params = json.dumps({
        "proposal": proposal,
        "tdr_info": tdr_info,
        "specific_requirements": tdr_requirements
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
        
        # Generar recomendaciones de mejora específicas
        improvement_recommendations = generate_improvement_recommendations(proposal, evaluation, tdr_info)
        state["improvement_recommendations"] = improvement_recommendations
        
    try:
        # Intentar convertir la evaluación a diccionario para presentarla mejor
        eval_dict = json.loads(evaluation)
        status = eval_dict.get("status", "desconocido")
        puntuacion = eval_dict.get("puntuacion", "N/A")
        
        # Preparar mensaje de resumen de evaluación detallado
        fortalezas = "\n".join([f"- {item}" for item in eval_dict.get("fortalezas", ["No se identificaron fortalezas específicas."])[:3]])
        debilidades = "\n".join([f"- {item}" for item in eval_dict.get("debilidades", ["No se identificaron debilidades específicas."])[:3]])
        
        # Calcular estadísticas de cumplimiento
        cumplimiento = eval_dict.get("cumplimiento_requisitos", {})
        total_reqs = len(cumplimiento)
        cumplidos = sum(1 for val in cumplimiento.values() if val)
        porcentaje = int((cumplidos / total_reqs) * 100) if total_reqs > 0 else 0
        
        # Actualizar mensaje en el historial con evaluación detallada
        eval_message = (
            f"Propuesta evaluada: Estado '{status}' con puntuación {puntuacion}/10.\n\n"
            f"📊 **Resumen de evaluación:**\n"
            f"• Cumplimiento: {porcentaje}% ({cumplidos}/{total_reqs} requisitos)\n\n"
            f"💪 **Principales fortalezas:**\n{fortalezas}\n\n"
            f"🔍 **Áreas de mejora:**\n{debilidades}\n\n"
        )
        
        # Añadir recomendaciones si existen
        if "improvement_recommendations" in state:
            recomendaciones = state["improvement_recommendations"]
            eval_message += f"📌 **Recomendaciones específicas:**\n{recomendaciones}\n\n"
        
        # Añadir información del archivo guardado
        eval_message += f"La propuesta ha sido completada y guardada en el archivo '{state.get('proposal_filename', 'propuesta.txt')}'."
        
        state["messages"].append(AIMessage(content=eval_message))
    except (json.JSONDecodeError, TypeError):
        # Si hay error al decodificar el JSON, usar texto plano
        state["messages"].append(AIMessage(
            content=f"Evaluación completada. La propuesta ha sido guardada en el archivo '{state.get('proposal_filename', 'propuesta.txt')}'."
        ))
    
    # Siguiente paso (fin del proceso)
    state["next_step"] = "end"
    
    logger.info(f"Estado después de evaluate_proposal_node: {format_state_for_log(state)}")
    return state

def extract_tdr_requirements(tdr_info: str) -> dict:
    """
    Extrae requisitos específicos del TDR para la evaluación.
    
    Args:
        tdr_info: Información extraída del TDR
        
    Returns:
        Diccionario con requisitos específicos
    """
    requirements = {
        "tecnologias_especificas": [],
        "plazos_criticos": [],
        "entregables_obligatorios": [],
        "requisitos_especiales": []
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
            # Extraer tecnologías específicas
            tech_fields = ["tecnologias", "requisitos_tecnicos", "herramientas"]
            for field in tech_fields:
                if field in tdr_dict and tdr_dict[field]:
                    if isinstance(tdr_dict[field], list):
                        requirements["tecnologias_especificas"].extend(tdr_dict[field])
                    elif isinstance(tdr_dict[field], str):
                        # Intentar extraer elementos de una lista en texto
                        items = re.findall(r'[-•*]\s+([^-•*\n]+)', tdr_dict[field])
                        if items:
                            requirements["tecnologias_especificas"].extend(items)
                        else:
                            # Si no se encuentran viñetas, añadir el texto completo
                            requirements["tecnologias_especificas"].append(tdr_dict[field])
            
            # Extraer plazos críticos
            time_fields = ["plazos", "cronograma", "fechas", "fecha_entrega"]
            for field in time_fields:
                if field in tdr_dict and tdr_dict[field]:
                    if isinstance(tdr_dict[field], list):
                        requirements["plazos_criticos"].extend(tdr_dict[field])
                    elif isinstance(tdr_dict[field], str):
                        requirements["plazos_criticos"].append(tdr_dict[field])
            
            # Extraer entregables obligatorios
            deliv_fields = ["entregables", "productos", "deliverables"]
            for field in deliv_fields:
                if field in tdr_dict and tdr_dict[field]:
                    if isinstance(tdr_dict[field], list):
                        requirements["entregables_obligatorios"].extend(tdr_dict[field])
                    elif isinstance(tdr_dict[field], str):
                        items = re.findall(r'[-•*]\s+([^-•*\n]+)', tdr_dict[field])
                        if items:
                            requirements["entregables_obligatorios"].extend(items)
                        else:
                            requirements["entregables_obligatorios"].append(tdr_dict[field])
            
            # Extraer requisitos especiales
            special_fields = ["requisitos_especiales", "consideraciones", "restricciones"]
            for field in special_fields:
                if field in tdr_dict and tdr_dict[field]:
                    if isinstance(tdr_dict[field], list):
                        requirements["requisitos_especiales"].extend(tdr_dict[field])
                    elif isinstance(tdr_dict[field], str):
                        requirements["requisitos_especiales"].append(tdr_dict[field])
        else:
            # Si no es JSON, extraer mediante LLM
            requirements = extract_requirements_with_llm(tdr_info)
    except Exception as e:
        logger.warning(f"Error al extraer requisitos del TDR: {str(e)}")
    
    return requirements

def extract_requirements_with_llm(tdr_info: str) -> dict:
    """
    Utiliza el LLM para extraer requisitos específicos del TDR.
    
    Args:
        tdr_info: Información del TDR
        
    Returns:
        Diccionario con requisitos específicos
    """
    try:
        from llm.model import get_llm
        llm = get_llm()
        
        prompt = (
            "Analiza el siguiente TDR (Términos de Referencia) y extrae específicamente:\n"
            "1. Tecnologías específicas mencionadas\n"
            "2. Plazos críticos o fechas importantes\n"
            "3. Entregables obligatorios\n"
            "4. Requisitos especiales o consideraciones particulares\n\n"
            "Responde ÚNICAMENTE en formato JSON con estas cuatro categorías como claves "
            "y listas de elementos específicos como valores. Si no encuentras información "
            "para alguna categoría, devuelve una lista vacía.\n\n"
            f"TDR:\n{tdr_info[:3000]}"
        )
        
        response = llm.invoke(prompt)
        
        # Intentar extraer JSON
        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            requirements = json.loads(json_str)
            return requirements
        else:
            return {
                "tecnologias_especificas": [],
                "plazos_criticos": [],
                "entregables_obligatorios": [],
                "requisitos_especiales": []
            }
    except Exception as e:
        logger.warning(f"Error al extraer requisitos con LLM: {str(e)}")
        return {
            "tecnologias_especificas": [],
            "plazos_criticos": [],
            "entregables_obligatorios": [],
            "requisitos_especiales": []
        }

def generate_improvement_recommendations(proposal: str, evaluation: str, tdr_info: str) -> str:
    """
    Genera recomendaciones específicas para mejorar la propuesta.
    
    Args:
        proposal: Texto de la propuesta
        evaluation: Evaluación de la propuesta
        tdr_info: Información del TDR
        
    Returns:
        Texto con recomendaciones específicas
    """
    try:
        # Extraer debilidades de la evaluación
        eval_dict = json.loads(evaluation)
        debilidades = eval_dict.get("debilidades", [])
        
        if not debilidades:
            return "No se identificaron áreas específicas que requieran mejoras."
        
        # Utilizar las principales debilidades para generar recomendaciones
        top_debilidades = debilidades[:3]  # Limitar a las 3 principales
        
        # Crear recomendaciones a partir de las debilidades
        recommendations = []
        for i, debilidad in enumerate(top_debilidades):
            recommendation = f"- {debilidad.replace('Falta de', 'Incluir').replace('No se', 'Se recomienda')}"
            recommendations.append(recommendation)
        
        return "\n".join(recommendations)
    except Exception as e:
        logger.warning(f"Error al generar recomendaciones: {str(e)}")
        return "No se pudieron generar recomendaciones específicas."