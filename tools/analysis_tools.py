#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para análisis del TDR y generación de estructura.
"""

import json
import logging
import re
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path
from llm.model import get_llm

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

@tool
def analyze_tdr(text: str) -> str:
    """
    Usa el modelo LLM para extraer información clave del TDR.
    
    Args:
        text: Texto del TDR
        
    Returns:
        Información extraída del TDR en formato JSON
    """
    logger.info("Analizando el TDR")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "analyze_tdr",
        "Extrayendo información clave del TDR"
    )
    
    try:
        llm = get_llm()
        prompt = (
            "Extrae la siguiente información del TDR (Términos de Referencia):\n"
            "- Título del proyecto\n"
            "- Cliente o entidad solicitante\n"
            "- Alcance del proyecto\n"
            "- Lista de entregables\n"
            "- Materiales y equipos requeridos\n"
            "- Lista de actividades\n"
            "- Normativas mencionadas\n"
            "- Plazos y cronogramas\n"
            "- Requisitos técnicos específicos\n"
            "- Criterios de evaluación\n\n"
            "Responde en formato JSON con claves en español. Sé específico y concreto.\n\n"
            f"TDR:\n{text}"
        )
        
        response = llm.invoke(prompt)
        
        # Limpiar la respuesta si contiene delimitadores de código
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        logger.info(f"Análisis del TDR completado: {response[:200]}...")
        
        # Registrar éxito
        add_to_execution_path(
            "analyze_tdr_result",
            "Análisis completado exitosamente"
        )
        
        return response
    except Exception as e:
        error_message = f"Error al analizar el TDR: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "analyze_tdr_error",
            error_message
        )
        
        return f"Error: {error_message}"

@tool
def generate_index(info: str) -> str:
    """
    Genera un índice detallado para la propuesta técnica basado en la información extraída.
    
    Args:
        info: Información extraída del TDR
        
    Returns:
        Índice en formato JSON
    """
    logger.info("Generando índice detallado para la propuesta técnica")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "generate_index",
        "Generando estructura e índice completo de la propuesta"
    )
    
    try:
        llm = get_llm()
        prompt = (
            "Con la siguiente información extraída del TDR, genera un índice DETALLADO para la propuesta técnica. "
            "El índice debe incluir TODAS las secciones requeridas según el documento PKS-537 RQ-01, "
            "adaptadas específicamente al proyecto descrito en el TDR.\n\n"
            "Cada sección debe ser una clave en el JSON, con una descripción ESPECÍFICA de lo que debe contener, "
            "NO genérica. Las descripciones deben hacer referencia directa a elementos concretos mencionados en el TDR.\n\n"
            "Las secciones OBLIGATORIAS que debe incluir son:\n"
            "1. INTRODUCCIÓN Y CONTEXTO: Contexto específico del proyecto y problema a resolver\n"
            "2. OBJETIVOS: Objetivo general y objetivos específicos claramente definidos\n"
            "3. ALCANCE DEL TRABAJO: Descripción detallada de lo que incluye y no incluye el proyecto\n"
            "4. METODOLOGÍA PROPUESTA: Enfoque metodológico específico y justificado\n"
            "5. PLAN DE TRABAJO Y CRONOGRAMA: Fases, actividades y tiempos concretos\n"
            "6. ENTREGABLES: Listado detallado de productos a entregar\n"
            "7. RECURSOS HUMANOS Y TÉCNICOS: Equipo de trabajo y recursos necesarios\n"
            "8. GESTIÓN DE RIESGOS: Identificación y mitigación de riesgos específicos\n"
            "9. PLAN DE CALIDAD: Aseguramiento de calidad y estándares aplicables\n"
            "10. NORMATIVAS Y ESTÁNDARES: Cumplimiento de regulaciones relevantes\n"
            "11. EXPERIENCIA RELEVANTE: Proyectos similares y capacidades demostradas\n"
            "12. ANEXOS TÉCNICOS: Documentación complementaria\n\n"
            "Devuelve la respuesta ÚNICAMENTE en formato JSON válido sin texto adicional, sin tags <think>, "
            "sin comentarios y sin explicaciones.\n\n"
            "Información del TDR:\n"
            f"{info}\n\n"
            "Formato de respuesta JSON (ejemplo):\n"
            "{\n"
            "  \"INTRODUCCION_Y_CONTEXTO\": \"Descripción específica adaptada al proyecto...\",\n"
            "  \"OBJETIVOS_GENERALES\": \"Descripción específica...\",\n"
            "  ...\n"
            "}\n\n"
            "IMPORTANTE: Asegúrate de que las descripciones sean ESPECÍFICAS al proyecto descrito en el TDR "
            "y no frases genéricas que podrían aplicarse a cualquier proyecto."
        )
        
        response = llm.invoke(prompt)

        # Limpieza agresiva para tener un JSON válido
        # 1. Eliminar tags <think> y </think>
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 2. Eliminar cualquier texto antes de la primera llave de apertura
        response = re.sub(r'^[^{]*', '', response)
        
        # 3. Eliminar cualquier texto después de la última llave de cierre
        response = re.sub(r'[^}]*$', '', response)
        
        # 4. Limpiar la respuesta si contiene delimitadores de código
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        # Validar que sea un JSON válido
        try:
            json_response = json.loads(response)
            
            # Asegurar que incluye todas las secciones obligatorias
            required_sections = [
                "INTRODUCCION", "OBJETIVOS", "ALCANCE", "METODOLOGIA", 
                "PLAN_DE_TRABAJO", "ENTREGABLES", "RECURSOS", 
                "RIESGOS", "CALIDAD", "NORMATIVAS", "EXPERIENCIA", "ANEXOS"
            ]
            
            # Verificar que las secciones obligatorias estén presentes (con coincidencia parcial)
            missing_sections = []
            for required in required_sections:
                found = False
                for key in json_response.keys():
                    if required in key.upper().replace(" ", "_"):
                        found = True
                        break
                if not found:
                    missing_sections.append(required)
            
            # Si faltan secciones, añadirlas con descripciones predeterminadas
            if missing_sections:
                logger.warning(f"Faltan secciones obligatorias en el índice: {missing_sections}")
                
                default_descriptions = {
                    "INTRODUCCION": "Contexto del proyecto y descripción del problema a resolver",
                    "OBJETIVOS": "Objetivo general y objetivos específicos del proyecto",
                    "ALCANCE": "Alcance detallado del trabajo y límites del proyecto",
                    "METODOLOGIA": "Metodología propuesta para el desarrollo del proyecto",
                    "PLAN_DE_TRABAJO": "Plan de trabajo y cronograma de actividades",
                    "ENTREGABLES": "Listado detallado de entregables del proyecto",
                    "RECURSOS": "Recursos humanos y técnicos asignados al proyecto",
                    "RIESGOS": "Gestión de riesgos y plan de mitigación",
                    "CALIDAD": "Plan de aseguramiento de calidad",
                    "NORMATIVAS": "Normativas y estándares aplicables",
                    "EXPERIENCIA": "Experiencia relevante en proyectos similares",
                    "ANEXOS": "Anexos técnicos y documentación complementaria"
                }
                
                for section in missing_sections:
                    json_response[f"{section}_SECCION"] = default_descriptions[section]
            
            # Convertir de nuevo a JSON string
            response = json.dumps(json_response, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            logger.warning(f"La respuesta no es un JSON válido. Intentando reparar... Error: {str(e)}")
            
            # Intentar reparar JSON
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)
            
            # Si todo falla, crear un índice predeterminado
            try:
                json.loads(response)  # Intentar de nuevo después de la reparación
            except json.JSONDecodeError:
                logger.warning("Reparación fallida. Generando índice predeterminado.")
                
                # Crear un índice predeterminado más completo
                default_index = {
                    "INTRODUCCION_Y_CONTEXTO": "Contexto del proyecto y descripción del problema a resolver",
                    "OBJETIVOS_GENERALES": "Objetivo general del proyecto",
                    "OBJETIVOS_ESPECIFICOS": "Objetivos específicos detallados del proyecto",
                    "ALCANCE_DEL_TRABAJO": "Alcance detallado del trabajo, incluyendo lo que está dentro y fuera del alcance",
                    "METODOLOGIA_PROPUESTA": "Metodología propuesta para el desarrollo del proyecto",
                    "PLAN_DE_TRABAJO_Y_CRONOGRAMA": "Plan detallado con fases, actividades y tiempos",
                    "ENTREGABLES": "Listado y descripción detallada de los entregables del proyecto",
                    "RECURSOS_HUMANOS_Y_TECNICOS": "Equipo de trabajo, roles y recursos técnicos asignados",
                    "GESTION_DE_RIESGOS": "Identificación de riesgos y estrategias de mitigación",
                    "PLAN_DE_CALIDAD": "Mecanismos de aseguramiento de calidad y estándares aplicables",
                    "NORMATIVAS_Y_ESTANDARES_APLICABLES": "Cumplimiento de normativas y estándares relevantes",
                    "EXPERIENCIA_RELEVANTE": "Experiencia en proyectos similares y capacidades demostradas",
                    "ANEXOS_TECNICOS": "Documentación técnica complementaria"
                }
                
                response = json.dumps(default_index, ensure_ascii=False)
        
        logger.info(f"Índice generado: {response[:200]}...")
        
        # Registrar éxito
        add_to_execution_path(
            "generate_index_result",
            "Índice generado exitosamente con todas las secciones requeridas"
        )
        
        return response
        
    except Exception as e:
        error_message = f"Error al generar índice: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "generate_index_error",
            error_message
        )
        
        # Si hay un error, generamos un índice predeterminado en lugar de fallar
        default_index = {
            "INTRODUCCION_Y_CONTEXTO": "Contexto del proyecto y descripción del problema a resolver",
            "OBJETIVOS_GENERALES": "Objetivo general del proyecto",
            "OBJETIVOS_ESPECIFICOS": "Objetivos específicos detallados del proyecto",
            "ALCANCE_DEL_TRABAJO": "Alcance detallado del trabajo, incluyendo lo que está dentro y fuera del alcance",
            "METODOLOGIA_PROPUESTA": "Metodología propuesta para el desarrollo del proyecto",
            "PLAN_DE_TRABAJO_Y_CRONOGRAMA": "Plan detallado con fases, actividades y tiempos",
            "ENTREGABLES": "Listado y descripción detallada de los entregables del proyecto",
            "RECURSOS_HUMANOS_Y_TECNICOS": "Equipo de trabajo, roles y recursos técnicos asignados",
            "GESTION_DE_RIESGOS": "Identificación de riesgos y estrategias de mitigación",
            "PLAN_DE_CALIDAD": "Mecanismos de aseguramiento de calidad y estándares aplicables",
            "NORMATIVAS_Y_ESTANDARES_APLICABLES": "Cumplimiento de normativas y estándares relevantes",
            "EXPERIENCIA_RELEVANTE": "Experiencia en proyectos similares y capacidades demostradas",
            "ANEXOS_TECNICOS": "Documentación técnica complementaria"
        }
        
        logger.info("Generando índice predeterminado debido al error")
        return json.dumps(default_index, ensure_ascii=False)