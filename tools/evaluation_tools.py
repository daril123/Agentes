#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para evaluación y guardado de la propuesta técnica.
"""
import re
import json
import logging
import os
import glob
from datetime import datetime
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path
from llm.model import get_llm

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")



filename = None
@tool
def evaluate_proposal(params_str: str) -> str:
    """
    Evalúa la propuesta final para validar que cumpla con lo solicitado.
    
    Args:
        params_str: String en formato JSON con:
                 - proposal: Texto de la propuesta técnica
                 - tdr_info: Información extraída del TDR
        
    Returns:
        Evaluación en formato JSON
    """
    logger.info("Evaluando la propuesta técnica")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "evaluate_proposal",
        "Verificando calidad y alineación con los requisitos"
    )
    
    try:
        # Parsear los parámetros JSON
        params = json.loads(params_str)
        proposal = params.get("proposal", "")
        tdr_info = params.get("tdr_info", "")
        
        llm = get_llm()
        prompt = (
            "Evalúa la siguiente propuesta técnica y verifica que cumpla con los requisitos específicos del documento PKS-537 RQ-01. "
            "La propuesta debe incluir obligatoriamente estas secciones:\n"
            "1. Introducción y contexto del proyecto\n"
            "2. Objetivos (general y específicos)\n"
            "3. Alcance detallado del trabajo\n"
            "4. Metodología propuesta\n"
            "5. Plan de trabajo y cronograma\n"
            "6. Entregables con descripción detallada\n"
            "7. Recursos humanos y técnicos asignados\n"
            "8. Gestión de riesgos\n"
            "9. Plan de calidad\n"
            "10. Normativas y estándares aplicables\n"
            "11. Experiencia relevante en proyectos similares\n"
            "12. Anexos técnicos\n\n"
            "Verifica que cada sección tenga suficiente detalle y cumpla con los criterios de calidad esperados.\n\n"
            "Información del TDR:\n"
            f"{tdr_info}\n\n"
            "Propuesta Técnica (parcial):\n"
            f"{proposal[:5000]}...\n\n"  # Limitamos a los primeros 5000 caracteres
            "Responde en formato JSON con los siguientes campos:\n"
            "1. status: 'aprobado', 'aprobado_con_cambios', o 'rechazado'\n"
            "2. puntuacion: valor numérico del 1 al 10\n"
            "3. fortalezas: lista de puntos fuertes\n"
            "4. debilidades: lista de puntos débiles\n"
            "5. recomendaciones: lista de mejoras sugeridas\n"
            "6. cumplimiento_requisitos: un objeto con cada sección requerida como clave y un booleano indicando si se cumple adecuadamente"
        )
        
        response = llm.invoke(prompt)
        
        # Limpiar la respuesta JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        logger.info(f"Evaluación completada: {response[:200]}...")
        
        # Registrar éxito
        add_to_execution_path(
            "evaluate_proposal_result",
            "Evaluación completada exitosamente"
        )
        
        return response
    except Exception as e:
        error_message = f"Error al evaluar propuesta: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "evaluate_proposal_error",
            error_message
        )
        
        return f"Error: {error_message}"

@tool
def save_proposal_to_txt(params_str: str) -> str:
    """
    Guarda la propuesta final en un archivo TXT.
    
    Args:
        params_str: String o JSON con la propuesta y opciones
        
    Returns:
        Ruta del archivo guardado
    """
    # Extraer propuesta y parámetros
    global filename 
    filename = None
    tdr_name = None
    
    try:
        # Intentar parsear como JSON
        if isinstance(params_str, str):
            try:
                params = json.loads(params_str)
                proposal = params.get("proposal", "")
                tdr_name = params.get("tdr_name")
                filename = params.get("filename")
            except json.JSONDecodeError:
                # Si no es JSON, asumir que es directamente la propuesta
                proposal = params_str
        else:
            proposal = params_str
    except Exception as e:
        return f"Error: No se pudo procesar los parámetros: {str(e)}"
    
    # Extraer un nombre para el TDR si no se proporciona
    if not tdr_name:
        # Intenta extraer un título/nombre del TDR desde la propuesta
        import re
        title_match = re.search(r"Título del proyecto[:\s]+(.*?)(?:\n|$)", proposal, re.IGNORECASE)
        if title_match:
            tdr_name = title_match.group(1).strip()[:50]  # Limitar longitud
        else:
            tdr_name = "SinNombre"
    
    # Sanitizar el nombre del TDR
    import re 
    tdr_name = re.sub(r'[^\w\s-]', '', tdr_name)
    tdr_name = re.sub(r'[\s]+', '_', tdr_name)
    
    # Generar nombre de archivo
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Usar un nombre de archivo predecible
    if not filename:
        filename = f"Propuesta_{tdr_name}_{current_date}.txt"
    
    # Asegurar que estamos trabajando con solo el nombre del archivo, no la ruta
    base_filename = os.path.basename(filename)
    
    logger.info(f"Guardando propuesta en archivo: {base_filename}")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "save_proposal",
        f"Guardando en archivo: {base_filename}"
    )
    
    try:
        # Asegurar que el directorio de salida exista - CORRECCIÓN AQUÍ
        output_dir = "propuestas"
        
        # Crear el directorio de forma segura con manejo de errores
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Directorio creado o verificado: {output_dir}")
        except Exception as e:
            logger.warning(f"No se pudo crear el directorio {output_dir}: {str(e)}")
            # Usar el directorio actual como alternativa
            output_dir = "."
            logger.info(f"Usando directorio actual como alternativa")
        
        # Usar os.path.join para formar rutas de manera segura en diferentes sistemas operativos
        file_path = os.path.join(output_dir, base_filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(proposal)
        
        logger.info(f"Propuesta guardada exitosamente: {file_path}")
        
        # Registrar éxito
        add_to_execution_path(
            "save_proposal_result",
            f"Archivo guardado: {file_path}"
        )
        
        return file_path
    except Exception as e:
        error_message = f"Error al guardar propuesta: {str(e)}"
        logger.error(error_message)
        
        # Intentar guardar en el directorio actual como plan B
        try:
            backup_path = base_filename  # Solo el nombre del archivo en el directorio actual
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(proposal)
            
            logger.info(f"Propuesta guardada en ubicación alternativa: {backup_path}")
            
            add_to_execution_path(
                "save_proposal_result",
                f"Archivo guardado en ubicación alternativa: {backup_path}"
            )
            
            return backup_path
        except Exception as e2:
            logger.error(f"Error al guardar propuesta en ubicación alternativa: {str(e2)}")
            
            # Registrar error
            add_to_execution_path(
                "save_proposal_error",
                error_message
            )
            
            return f"Error: {error_message}"
@tool
def generate_proposal_summary(evaluation: str) -> str:
    """
    Genera un resumen ejecutivo de la propuesta basado en la evaluación.
    
    Args:
        evaluation: Evaluación de la propuesta en formato JSON
        
    Returns:
        Texto con el resumen ejecutivo
    """
    logger.info("Generando resumen ejecutivo de la propuesta")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "generate_summary",
        "Creando resumen ejecutivo de la propuesta"
    )
    
    try:
        # Parsear la evaluación
        eval_dict = json.loads(evaluation)
        
        llm = get_llm()
        prompt = (
            "Genera un resumen ejecutivo breve para la propuesta técnica basado en su evaluación. "
            "El resumen debe destacar los puntos fuertes, mencionar áreas de mejora y dar una valoración global. "
            "Debe ser profesional y objetivo.\n\n"
            f"Evaluación:\n{json.dumps(eval_dict, indent=2)}\n\n"
            "El resumen no debe exceder los 500 caracteres."
        )
        
        summary = llm.invoke(prompt)
        
        logger.info(f"Resumen generado: {summary[:100]}...")
        
        # Registrar éxito
        add_to_execution_path(
            "generate_summary_result",
            "Resumen ejecutivo generado exitosamente"
        )
        
        return summary
    except Exception as e:
        error_message = f"Error al generar resumen: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "generate_summary_error",
            error_message
        )
        
        return f"Error: {error_message}"