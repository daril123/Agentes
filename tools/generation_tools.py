#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para generación de secciones y combinación de la propuesta técnica.
"""

import json
import logging
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path
from llm.model import get_llm
import re

from tools.crag_tools import get_similar_proposals_context as crag_get_similar_proposals_context

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

@tool
def generate_section(params: str) -> str:
    """
    Redacta una sección específica de la propuesta técnica.
    
    Args:
        params: String en formato JSON con los siguientes campos:
               - section_name: Nombre de la sección
               - description: Descripción de la sección del índice
               - info: Información extraída del TDR
               - previous_content: Contenido previamente generado (opcional)
        
    Returns:
        Texto de la sección generada
    """
    try:
        # Parsear los parámetros JSON
        params_dict = json.loads(params)
        section_name = params_dict.get("section_name", "")
        description = params_dict.get("description", "")
        info = params_dict.get("info", "")
        previous_content = params_dict.get("previous_content", "")
        
        logger.info(f"Generando sección: {section_name}")
        
        # Registrar en el historial de ejecución
        add_to_execution_path(
            "generate_section",
            f"Redactando sección: {section_name}"
        )
        
        # Buscar propuestas similares para obtener contexto
        context = get_similar_proposals_context(section_name, info)
        
        # Guía específica según el tipo de sección (según PKS-537 RQ-01)
        section_guidance = {
            "Introducción": "Proporciona contexto del proyecto, antecedentes y justificación. Describe brevemente el problema que se va a resolver.",
            "Objetivos": "Incluye tanto el objetivo general como los objetivos específicos. Estos deben ser medibles y estar alineados con el alcance del proyecto.",
            "Alcance": "Detalla claramente los límites del proyecto, qué se incluye y qué no. Enumera de forma precisa los componentes, sistemas o áreas que abarcará el trabajo.",
            "Metodología": "Describe el enfoque técnico que se utilizará, marcos de trabajo (frameworks), métodos y técnicas específicas. Justifica por qué este enfoque es adecuado.",
            "Plan de trabajo": "Presenta un cronograma detallado con fases, actividades, duración estimada y dependencias. Incluye hitos clave y puntos de control.",
            "Entregables": "Enumera todos los productos a entregar con descripciones detalladas, formato, contenido y criterios de aceptación para cada uno.",
            "Recursos": "Detalla el equipo humano (roles, perfiles, responsabilidades) y recursos técnicos (equipamiento, software, infraestructura) que se asignarán al proyecto.",
            "Riesgos": "Identifica posibles riesgos, evalúa su impacto y probabilidad, y propone estrategias de mitigación específicas para cada uno.",
            "Calidad": "Describe los procesos, estándares y métricas que se utilizarán para asegurar la calidad. Incluye procedimientos de verificación y validación.",
            "Normativas": "Enumera todas las normativas, estándares y regulaciones aplicables al proyecto y explica cómo se asegurará su cumplimiento.",
            "Experiencia": "Presenta proyectos similares realizados anteriormente, destacando aspectos clave, resultados obtenidos y lecciones aprendidas relevantes para este proyecto.",
            "Anexos": "Incluye información técnica adicional, diagramas, especificaciones detalladas u otra documentación relevante que respalde la propuesta."
        }
        
        # Buscar guía específica para la sección actual
        specific_guidance = ""
        for key, guide in section_guidance.items():
            if key.lower() in section_name.lower():
                specific_guidance = f"Guía específica para esta sección: {guide}\n\n"
                break
        
        llm = get_llm()
        prompt = (
            f"Redacta la sección de '{section_name}' para la propuesta técnica. "
            "Utiliza la siguiente descripción como guía:\n"
            f"{description}\n\n"
            f"{specific_guidance}"
            "Adicionalmente, toma en cuenta la siguiente información extraída del TDR:\n"
            f"{info}\n\n"
        )
        
        # Añadir contexto de contenido previo si existe
        if previous_content:
            prompt += "### Contenido previo de la propuesta:\n"
            prompt += previous_content + "\n\n"
            prompt += "Asegúrate de que tu sección sea coherente con el contenido anterior "
            prompt += "y evita repetir información que ya se haya cubierto. "
            prompt += "Mantén una estructura y estilo consistentes con las secciones anteriores.\n\n"
        
        # Añadir contexto de propuestas similares si existe
        if context:
            prompt += "### Contexto de propuestas similares anteriores:\n"
            prompt += context + "\n\n"
            prompt += "Usa este contexto como inspiración, pero personaliza el contenido para este proyecto específico.\n\n"
        
        prompt += (
            "Responde con el texto de la sección en formato profesional y detallado. "
            "Usa lenguaje técnico apropiado y estructura el contenido en párrafos claros. "
            "La extensión debe ser proporcional a la importancia de la sección. "
            "Asegúrate de cumplir con todos los requisitos del documento proporcionado para esta sección. "
            "Incluye subsecciones numeradas cuando sea apropiado y utiliza viñetas para listas."
        )
        
        response = llm.invoke(prompt)
        
        logger.info(f"Sección '{section_name}' generada: {len(response)} caracteres")
        
        # Registrar éxito
        add_to_execution_path(
            "generate_section_result",
            f"Sección {section_name} generada ({len(response)} caracteres)"
        )
        
        return f"## {section_name}\n\n{response}"
    except Exception as e:
        section_name = json.loads(params).get("section_name", "desconocida") if isinstance(params, str) else "desconocida"
        error_message = f"Error al generar sección {section_name}: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "generate_section_error",
            error_message
        )
        
        return f"Error en sección {section_name}: {error_message}"

def get_similar_proposals_context(section_name: str, tdr_info: str) -> str:
    """
    Obtiene contexto de propuestas similares para la sección actual.
    
    Args:
        section_name: Nombre de la sección
        tdr_info: Información extraída del TDR
        
    Returns:
        Texto con contexto de propuestas similares o cadena vacía si no hay
    """
    return crag_get_similar_proposals_context(section_name, tdr_info)

@tool
def combine_sections(params_str: str) -> str:
    """
    Combina todas las secciones en un solo documento de propuesta.
    
    Args:
        params_str: String en formato JSON con una lista de secciones
        
    Returns:
        Propuesta técnica completa
    """
    logger.info("Combinando secciones en propuesta final")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "combine_sections",
        "Integrando todas las secciones en un documento final"
    )
    
    try:
        # Parsear los parámetros JSON
        params = json.loads(params_str)
        sections = params.get("sections", [])
        
        # Añadir encabezado y título principal
        current_date = "25 de marzo, 2025"  # Puedes obtener la fecha actual
        proposal = f"""# PROPUESTA TÉCNICA
**Documento: PKS-537 RQ-01**
**Fecha: {current_date}**

---

"""
        
        # Añadir tabla de contenido
        proposal += "## Tabla de Contenido\n\n"
        section_pattern = r"## ([^\n]+)"
        toc_items = []
        
        for i, section in enumerate(sections):
            section_matches = re.findall(section_pattern, section)
            if section_matches:
                section_title = section_matches[0]
                toc_items.append(f"{i+1}. {section_title}")
                
        proposal += "\n".join(toc_items) + "\n\n---\n\n"
        
        # Combinar secciones
        proposal += "\n\n".join(sections)
        
        # Añadir sección final y número de versión
        proposal += "\n\n---\n\n**Versión 1.0**\n\n"
        proposal += "**Documento preparado en conformidad con los requisitos del documento PKS-537 RQ-01**"
        
        logger.info(f"Propuesta combinada: {len(proposal)} caracteres")
        
        # Registrar éxito
        add_to_execution_path(
            "combine_sections_result",
            f"Propuesta generada: {len(proposal)} caracteres"
        )
        
        return proposal
    except Exception as e:
        error_message = f"Error al combinar secciones: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "combine_sections_error",
            error_message
        )
        
        return f"Error: {error_message}"