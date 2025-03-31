#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para generación de secciones y combinación de la propuesta técnica.
"""

import json
import logging
import re
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path
from llm.model import get_llm

from tools.crag_tools import get_similar_proposals_context as crag_get_similar_proposals_context

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

@tool
def generate_section(params: str) -> str:
    """
    Redacta una sección específica de la propuesta técnica con contenido preciso y detallado.
    
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
        tdr_info = params_dict.get("info", "")
        previous_content = params_dict.get("previous_content", "")
        
        logger.info(f"Generando sección específica: {section_name}")
        
        # Registrar en el historial de ejecución
        add_to_execution_path(
            "generate_section",
            f"Redactando sección detallada: {section_name}"
        )
        
        # Buscar propuestas similares para obtener contexto
        context = get_similar_proposals_context(section_name, tdr_info)
        
        # Extraer información relevante para esta sección específica
        section_specific_info = extract_section_specific_info(section_name, tdr_info)
        
        # Crear prompt mejorado para generar contenido específico
        prompt = (
            f"Genera la sección '{section_name}' para una propuesta técnica profesional. "
            f"Esta sección debe ser ALTAMENTE ESPECÍFICA, basada estrictamente en el TDR, "
            f"evitando generalidades y lenguaje ambiguo.\n\n"
            
            f"DESCRIPCIÓN DE LA SECCIÓN:\n{description}\n\n"
            
            f"INFORMACIÓN ESPECÍFICA RELEVANTE DEL TDR:\n{section_specific_info}\n\n"
            
            f"TDR COMPLETO (referencia):\n{tdr_info[:1000]}...\n\n"
        )
        
        # Añadir contexto de contenido previo si existe
        if previous_content:
            prompt += "CONTENIDO PREVIO DE LA PROPUESTA:\n"
            prompt += previous_content + "\n\n"
            prompt += "Asegúrate de que tu sección sea coherente con el contenido anterior "
            prompt += "y evita repetir información que ya se haya cubierto. "
            prompt += "Mantén una estructura y estilo consistentes con las secciones anteriores.\n\n"
        
        # Añadir contexto de propuestas similares si existe
        if context:
            prompt += "EJEMPLOS DE PROPUESTAS SIMILARES (referencia):\n"
            prompt += context + "\n\n"
        
        # Añadir instrucciones específicas según la sección
        section_guidance = get_section_specific_guidance(section_name)
        prompt += f"GUÍA ESPECÍFICA PARA ESTA SECCIÓN:\n{section_guidance}\n\n"
        
        # Instrucciones críticas para especificidad
        prompt += (
            "INSTRUCCIONES CRÍTICAS:\n"
            "1. Sé EXTREMADAMENTE ESPECÍFICO y CONCRETO. Evita completamente las generalidades.\n"
            "2. Usa DATOS REALES extraídos del TDR, no inventes información.\n"
            "3. Incluye CIFRAS, MÉTRICAS Y DATOS TÉCNICOS precisos cuando sea posible.\n"
            "4. Para tablas, usa formato markdown con contenido específico y detallado.\n"
            "5. Para listas, usa viñetas con elementos concretos, no genéricos.\n"
            "6. Usa un lenguaje técnico y profesional en español.\n"
            "7. NO uses etiquetas <think> o similares.\n"
            "8. NO incluyas caracteres en otros idiomas.\n\n"
            
            "EVITA ABSOLUTAMENTE:\n"
            "- Frases genéricas como 'se implementará una metodología adecuada'\n"
            "- Texto ambiguo como 'varios recursos serán necesarios'\n"
            "- Menciones vagas como 'se seguirán las mejores prácticas'\n"
            "- Contenido que podría aplicarse a cualquier proyecto\n\n"
            
            "Responde con el texto completo de la sección en formato profesional y detallado."
        )
        
        # Generar la sección con el LLM
        llm = get_llm()
        response = llm.invoke(prompt)
        
        # Limpiar la respuesta
        response = clean_section_content(response)
        
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

def extract_section_specific_info(section_name: str, tdr_info: str) -> str:
    """
    Extrae información específica del TDR relevante para la sección.
    
    Args:
        section_name: Nombre de la sección
        tdr_info: Información extraída del TDR
        
    Returns:
        Información relevante para la sección específica
    """
    try:
        # Verificar si tdr_info es JSON
        tdr_dict = {}
        is_json = False
        try:
            tdr_dict = json.loads(tdr_info)
            is_json = True
        except json.JSONDecodeError:
            pass
        
        # Si no es JSON, usar el texto como está
        if not is_json:
            return tdr_info
        
        # Mapeo de secciones a campos relevantes
        section_fields = {
            "introduccion": ["titulo_proyecto", "cliente", "contexto", "descripcion", "problema"],
            "objetivos": ["objetivos", "metas", "proposito"],
            "alcance": ["alcance_proyecto", "limites", "requisitos_tecnicos"],
            "metodologia": ["metodologia", "enfoque", "procedimientos", "tecnologias"],
            "plan_trabajo": ["plazos", "cronograma", "actividades", "fases", "etapas"],
            "entregables": ["entregables", "productos", "deliverables"],
            "recursos": ["recursos", "personal", "equipo", "materiales_equipos"],
            "riesgos": ["riesgos", "amenazas", "contingencias"],
            "calidad": ["calidad", "estandares", "metricas"],
            "normativas": ["normativas", "regulaciones", "leyes", "estandares"],
            "experiencia": ["experiencia", "proyectos_similares"],
            "anexos": ["anexos", "documentacion_adicional"]
        }
        
        # Buscar la mejor coincidencia para la sección
        matched_section = None
        for key in section_fields.keys():
            if key in section_name.lower():
                matched_section = key
                break
        
        if not matched_section:
            # Si no hay coincidencia, devolver todos los campos
            return json.dumps(tdr_dict, indent=2)
        
        # Extraer campos relevantes
        relevant_fields = section_fields[matched_section]
        section_info = {}
        
        for field in tdr_dict.keys():
            for relevant in relevant_fields:
                if relevant in field.lower() and tdr_dict[field]:
                    section_info[field] = tdr_dict[field]
        
        if not section_info:
            return json.dumps(tdr_dict, indent=2)
        
        return json.dumps(section_info, indent=2)
    except Exception as e:
        logger.error(f"Error al extraer información específica: {str(e)}")
        return tdr_info

def get_section_specific_guidance(section_name: str) -> str:
    """
    Proporciona guía específica para cada tipo de sección.
    
    Args:
        section_name: Nombre de la sección
        
    Returns:
        Guía específica para la sección
    """
    section_lower = section_name.lower()
    
    guidance = {
        "introduccion": (
            "- Menciona específicamente el nombre del proyecto y el cliente\n"
            "- Describe el contexto real del sector donde se desarrolla\n"
            "- Explica el problema concreto que aborda el proyecto\n"
            "- Incluye datos específicos sobre la situación actual"
        ),
        "objetivos": (
            "- Formula objetivos SMART (Específicos, Medibles, Alcanzables, Relevantes, Temporales)\n"
            "- El objetivo general debe ser un párrafo concreto\n"
            "- Los objetivos específicos deben ser 3-5 puntos con verbos de acción\n"
            "- Incluye métricas o indicadores específicos para medir el éxito"
        ),
        "alcance": (
            "- Define claramente QUÉ SE INCLUYE y QUÉ NO SE INCLUYE en el proyecto\n"
            "- Especifica componentes, módulos o sistemas concretos a desarrollar\n"
            "- Menciona ubicaciones físicas específicas si aplica\n"
            "- Detalla las interfaces con otros sistemas existentes"
        ),
        "metodologia": (
            "- Nombra metodologías específicas (no solo términos genéricos como 'ágil')\n"
            "- Detalla fases concretas con actividades específicas en cada una\n"
            "- Menciona herramientas y técnicas específicas a utilizar\n"
            "- Explica cómo se abordarán los desafíos técnicos del proyecto"
        ),
        "plan_trabajo": (
            "- Proporciona un cronograma detallado con fechas o duraciones específicas\n"
            "- Desglosa tareas principales en subtareas concretas\n"
            "- Identifica dependencias entre tareas\n"
            "- Usa formato de tabla para mayor claridad"
        ),
        "entregables": (
            "- Nombra cada entregable con un título específico\n"
            "- Describe el formato exacto de cada entregable\n"
            "- Detalla el contenido específico de cada entregable\n"
            "- Establece criterios concretos de aceptación"
        ),
        "recursos": (
            "- Nombra roles específicos con sus responsabilidades\n"
            "- Especifica habilidades técnicas requeridas para cada rol\n"
            "- Detalla hardware y software específicos necesarios\n"
            "- Cuantifica recursos (horas-persona, capacidad, etc.)"
        ),
        "riesgos": (
            "- Identifica riesgos específicos del proyecto, no genéricos\n"
            "- Cuantifica la probabilidad e impacto de cada riesgo\n"
            "- Proporciona estrategias de mitigación concretas\n"
            "- Incluye planes de contingencia específicos"
        ),
        "calidad": (
            "- Nombra estándares específicos aplicables (ISO, IEEE, etc.)\n"
            "- Define métricas concretas de calidad con valores objetivo\n"
            "- Detalla procesos específicos de verificación\n"
            "- Especifica herramientas para pruebas y QA"
        ),
        "normativas": (
            "- Cita normativas específicas con sus códigos o referencias\n"
            "- Explica cómo se aplicará cada normativa al proyecto\n"
            "- Menciona certificaciones requeridas si aplica\n"
            "- Detalla procesos de conformidad y validación"
        ),
        "experiencia": (
            "- Menciona proyectos anteriores específicos\n"
            "- Incluye clientes reales o sectores específicos\n"
            "- Cuantifica resultados obtenidos en proyectos previos\n"
            "- Destaca tecnologías específicas utilizadas"
        ),
        "anexos": (
            "- Enumera documentos técnicos específicos a incluir\n"
            "- Describe el contenido de cada anexo\n"
            "- Menciona diagramas o modelos técnicos específicos\n"
            "- Referencia estándares utilizados en los anexos"
        )
    }
    
    # Buscar la mejor coincidencia
    for key, value in guidance.items():
        if key in section_lower:
            return value
    
    # Si no hay coincidencia, devolver guía genérica
    return (
        "- Sé extremadamente específico, evita generalidades\n"
        "- Usa datos concretos extraídos del TDR\n"
        "- Incluye detalles técnicos precisos\n"
        "- Estructura la información de forma clara y accesible"
    )

def clean_section_content(content: str) -> str:
    """
    Limpia el contenido de la sección para eliminar elementos no deseados.
    
    Args:
        content: Contenido a limpiar
        
    Returns:
        Contenido limpio
    """
    # Eliminar etiquetas think
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Eliminar caracteres no latinos (excepto puntuación común)
    content = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', content)
    
    # Eliminar líneas vacías múltiples
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Eliminar posibles encabezados que haya generado el LLM
    content = re.sub(r'^#+\s+.*\n', '', content, flags=re.MULTILINE)
    
    return content.strip()

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
        from datetime import datetime
        current_date = datetime.now().strftime("%d de %B, %Y")
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