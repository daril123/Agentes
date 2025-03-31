#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para procesamiento de archivos PDF.
"""

import re
import logging
import PyPDF2
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def limpiar_texto(texto):
    """
    Limpia el texto eliminando espacios en blanco innecesarios y saltos de línea duplicados.
    
    Args:
        texto: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    # Elimina espacios en blanco antes y después de cada salto de línea
    texto = re.sub(r'\s*\n\s*', '\n', texto)
    # Sustituye múltiples saltos de línea consecutivos por uno solo
    return re.sub(r'\n+', '\n', texto).strip()

@tool
def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extrae y concatena el texto de todas las páginas del PDF.
    
    Args:
        pdf_file_path: Ruta al archivo PDF
        
    Returns:
        Texto extraído del PDF
    """
    logger.info(f"Extrayendo texto del PDF: {pdf_file_path}")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "extract_text",
        f"Extrayendo texto del PDF: {pdf_file_path}"
    )
    
    try:
        text_content = []
        with open(pdf_file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_content.append(page.extract_text() or "")
        
        text_content = "\n".join(text_content)
        cleaned_text = limpiar_texto(text_content)
        
        logger.info(f"Texto extraído exitosamente: {len(cleaned_text)} caracteres")
        
        # Registrar éxito
        add_to_execution_path(
            "extract_text_result",
            f"Texto extraído: {len(cleaned_text)} caracteres"
        )
        
        return cleaned_text
    except Exception as e:
        error_message = f"Error al extraer texto: {str(e)}"
        logger.error(error_message)
        
        # Registrar error
        add_to_execution_path(
            "extract_text_error",
            error_message
        )
        
        return f"Error: {error_message}"
