#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mejoras al sistema de generación de secciones para propuestas técnicas.
"""

import json
import logging
import re
from langchain_core.tools import tool
from llm.model import get_llm
from core.execution_tracker import add_to_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

@tool
def generate_coherent_section(params: str) -> str:
    """
    Genera una sección coherente y estandarizada para la propuesta técnica.
    
    Args:
        params: String JSON con los siguientes campos:
               - section_name: Nombre de la sección
               - section_number: Número de la sección en la propuesta
               - description: Descripción de la sección
               - tdr_info: Información extraída del TDR
               - previous_sections: Lista de secciones anteriores
               - proposal_metadata: Metadatos de la propuesta (título, cliente, etc.)
        
    Returns:
        Texto de la sección generada con formato estandarizado
    """
    try:
        # Parsear los parámetros
        params_dict = json.loads(params)
        section_name = params_dict.get("section_name", "")
        section_number = params_dict.get("section_number", "")
        description = params_dict.get("description", "")
        tdr_info = params_dict.get("tdr_info", "")
        previous_sections = params_dict.get("previous_sections", [])
        metadata = params_dict.get("proposal_metadata", {})
        
        # Registrar inicio de generación
        logger.info(f"Generando sección coherente: {section_name} (#{section_number})")
        add_to_execution_path(
            "generate_coherent_section",
            f"Generando sección {section_name} con formato estandarizado"
        )
        
        # Crear template para formato consistente
        section_templates = {
            "introduccion": {
                "format": "## {number}. INTRODUCCIÓN\n\n{content}",
                "structure": [
                    "Contexto del proyecto",
                    "Descripción del problema/necesidad",
                    "Propósito de la propuesta",
                    "Alcance general"
                ]
            },
            "objetivos": {
                "format": "## {number}. OBJETIVOS\n\n### {number}.1 OBJETIVO GENERAL\n\n{general_objective}\n\n### {number}.2 OBJETIVOS ESPECÍFICOS\n\n{specific_objectives}",
                "structure": [
                    "Objetivo general claro y conciso",
                    "3-5 objetivos específicos en viñetas"
                ]
            },
            "alcance": {
                "format": "## {number}. ALCANCE DEL SERVICIO\n\n{content}",
                "structure": [
                    "Descripción detallada del alcance",
                    "Listado de actividades incluidas",
                    "Delimitación clara (qué incluye y qué no incluye)",
                    "Tablas de materiales o equipos si aplica"
                ]
            },
            "metodologia": {
                "format": "## {number}. METODOLOGÍA\n\n{content}",
                "structure": [
                    "Enfoque metodológico",
                    "Etapas o fases del trabajo",
                    "Técnicas y herramientas a utilizar",
                    "Procedimientos específicos"
                ]
            },
            "plan_trabajo": {
                "format": "## {number}. PLAN DE TRABAJO Y CRONOGRAMA\n\n{content}",
                "structure": [
                    "Descripción de etapas y actividades",
                    "Cronograma detallado",
                    "Hitos y entregables por etapa",
                    "Asignación de recursos por actividad"
                ]
            },
            "entregables": {
                "format": "## {number}. ENTREGABLES\n\n{content}",
                "structure": [
                    "Listado de entregables",
                    "Descripción detallada de cada uno",
                    "Formato y características",
                    "Cronograma de entrega"
                ]
            },
            "recursos": {
                "format": "## {number}. RECURSOS HUMANOS Y TÉCNICOS\n\n{content}",
                "structure": [
                    "Equipo de trabajo y roles",
                    "Perfiles profesionales",
                    "Equipamiento técnico",
                    "Recursos materiales"
                ]
            },
            "riesgos": {
                "format": "## {number}. GESTIÓN DE RIESGOS\n\n{content}",
                "structure": [
                    "Identificación de riesgos principales",
                    "Evaluación de impacto y probabilidad",
                    "Estrategias de mitigación",
                    "Plan de contingencia"
                ]
            },
            "plan_calidad": {
                "format": "## {number}. PLAN DE CALIDAD\n\n{content}",
                "structure": [
                    "Objetivos de calidad",
                    "Estándares aplicables",
                    "Procedimientos de aseguramiento",
                    "Controles y métricas"
                ]
            },
            "normativas": {
                "format": "## {number}. NORMATIVAS Y ESTÁNDARES APLICABLES\n\n{content}",
                "structure": [
                    "Listado de normativas aplicables",
                    "Estándares técnicos",
                    "Procedimientos de cumplimiento",
                    "Certificaciones relevantes"
                ]
            },
            "experiencia": {
                "format": "## {number}. EXPERIENCIA EN PROYECTOS SIMILARES\n\n{content}",
                "structure": [
                    "Proyectos similares ejecutados",
                    "Resultados obtenidos",
                    "Clientes principales",
                    "Lecciones aprendidas aplicables"
                ]
            },
            "anexos": {
                "format": "## {number}. ANEXOS TÉCNICOS\n\n{content}",
                "structure": [
                    "Listado de anexos",
                    "Información complementaria",
                    "Documentación técnica adicional"
                ]
            }
        }
        
        # Normalizar nombre de sección
        normalized_section = section_name.lower().strip()
        
        # Obtener template para la sección o usar uno genérico
        template = section_templates.get(normalized_section, {
            "format": f"## {{number}}. {section_name.upper()}\n\n{{content}}",
            "structure": ["Información general", "Detalles específicos"]
        })
        
        # Preparar contexto con secciones previas
        previous_context = ""
        if previous_sections:
            previous_context = "\n\n".join([
                f"Sección previa: {prev}" for prev in previous_sections[-2:]  # Últimas 2 secciones
            ])
        
        # Crear prompt con guía específica para la sección
        structure_guide = "\n".join([f"- {item}" for item in template["structure"]])
        
        prompt = f"""
        Genera la sección "{section_name}" (número {section_number}) para una propuesta técnica profesional.
        
        INFORMACIÓN DEL PROYECTO:
        - Título del proyecto: {metadata.get('titulo_proyecto', 'No especificado')}
        - Cliente: {metadata.get('cliente', 'No especificado')}
        - Descripción: {description}
        
        ESTRUCTURA RECOMENDADA:
        {structure_guide}
        
        INFORMACIÓN EXTRAÍDA DEL TDR:
        {tdr_info[:1000]}  # Limitar a 1000 caracteres
        
        CONTEXTO DE SECCIONES PREVIAS:
        {previous_context}
        
        INSTRUCCIONES IMPORTANTES:
        1. Genera SOLO el contenido, sin repetir el encabezado o título.
        2. Usa un lenguaje técnico y profesional en español.
        3. NO uses etiquetas <think> o similares.
        4. NO incluyas caracteres en otros idiomas (como chino o japonés).
        5. Sé específico y enfócate en los detalles técnicos relevantes.
        6. Mantén consistencia con las secciones previas.
        7. Organiza el contenido en párrafos claros y concisos.
        8. Usa viñetas o listas numeradas cuando sea apropiado.
        
        Tu respuesta se insertará directamente en el formato: {template["format"].format(number=section_number, content="[TU CONTENIDO AQUÍ]")}
        """
        
        # Generar el contenido con el LLM
        llm = get_llm()
        response = llm.invoke(prompt)
        
        # Limpiar la respuesta
        clean_response = clean_section_content(response)
        
        # Formatear según el template específico de la sección
        if normalized_section == "objetivos":
            # Extraer objetivo general y específicos
            general_pattern = r"(?:objetivo general|objetivo principal)[:\s]+(.*?)(?=objetivo|$)"
            specific_pattern = r"(?:objetivos específicos)[:\s]+(.*?)(?=$)"
            
            general_match = re.search(general_pattern, clean_response, re.IGNORECASE | re.DOTALL)
            specific_match = re.search(specific_pattern, clean_response, re.IGNORECASE | re.DOTALL)
            
            general_objective = general_match.group(1).strip() if general_match else clean_response
            specific_objectives = specific_match.group(1).strip() if specific_match else ""
            
            # Si no se encontraron objetivos específicos, buscar bullets o números
            if not specific_objectives:
                bullet_items = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', clean_response, re.DOTALL)
                numbered_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', clean_response, re.DOTALL)
                
                all_items = bullet_items or numbered_items
                if all_items:
                    specific_objectives = "\n".join([f"- {item.strip()}" for item in all_items])
            
            # Si aún no hay objetivos específicos, usar todo como general
            if not specific_objectives:
                general_objective = clean_response
                specific_objectives = "- Pendiente de definir objetivos específicos"
            
            formatted_section = template["format"].format(
                number=section_number,
                general_objective=general_objective,
                specific_objectives=specific_objectives
            )
        else:
            # Para otras secciones, usar el formato estándar
            formatted_section = template["format"].format(
                number=section_number,
                content=clean_response
            )
        
        logger.info(f"Sección {section_name} generada: {len(formatted_section)} caracteres")
        
        # Registrar éxito
        add_to_execution_path(
            "generate_coherent_section_result",
            f"Sección {section_name} generada exitosamente"
        )
        
        return formatted_section
        
    except Exception as e:
        error_message = f"Error al generar sección coherente {section_name}: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Registrar error
        add_to_execution_path(
            "generate_coherent_section_error",
            error_message
        )
        
        # Devolver una sección mínima en caso de error
        return f"## {section_number}. {section_name.upper()}\n\n[Esta sección no pudo generarse correctamente. Por favor, revise los logs para más detalles.]"

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
    
    # Eliminar posibles encabezados que haya generado el LLM
    content = re.sub(r'^#+\s+.*?\n', '', content, flags=re.MULTILINE)
    
    # Eliminar líneas vacías múltiples
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    return content.strip()

def validate_section_consistency(sections: list) -> dict:
    """
    Valida la consistencia entre secciones y sugiere correcciones.
    
    Args:
        sections: Lista de secciones generadas
        
    Returns:
        Diccionario con validación y sugerencias
    """
    issues = []
    suggestions = []
    
    # Verificar numeración consistente
    number_pattern = r'##\s+(\d+)\.'
    section_numbers = []
    
    for i, section in enumerate(sections):
        number_matches = re.findall(number_pattern, section)
        if number_matches:
            section_numbers.append(int(number_matches[0]))
        else:
            issues.append(f"La sección {i+1} no tiene numeración clara")
            suggestions.append(f"Añadir numeración '## {i+1}.' al inicio de la sección")
    
    # Verificar secuencia de números
    if section_numbers:
        expected_sequence = list(range(1, len(section_numbers) + 1))
        if section_numbers != expected_sequence:
            issues.append(f"Numeración inconsistente: {section_numbers} vs esperado {expected_sequence}")
            suggestions.append("Corregir la numeración para que sea secuencial (1, 2, 3...)")
    
    # Verificar idioma consistente
    non_latin_pattern = r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+'
    for i, section in enumerate(sections):
        if re.search(non_latin_pattern, section):
            issues.append(f"La sección {i+1} contiene caracteres no latinos")
            suggestions.append("Eliminar caracteres no latinos (p.ej., chinos, japoneses)")
    
    # Verificar formato consistente de subsecciones
    subsection_patterns = [r'###\s+\d+\.\d+', r'####\s+\d+\.\d+\.\d+']
    inconsistent_subsections = False
    
    for pattern in subsection_patterns:
        section_with_pattern = [i for i, s in enumerate(sections) if re.search(pattern, s)]
        if 0 < len(section_with_pattern) < len(sections) / 2:
            inconsistent_subsections = True
            
    if inconsistent_subsections:
        issues.append("Formato inconsistente de subsecciones entre diferentes partes del documento")
        suggestions.append("Estandarizar el uso de subsecciones (###) en todo el documento")
    
    return {
        "is_consistent": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions
    }