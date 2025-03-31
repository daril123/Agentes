#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para buscar y obtener contexto de propuestas técnicas anteriores.
"""

import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from core.execution_tracker import add_to_execution_path
from tools.crag_tools import find_similar_proposals as crag_find_similar_proposals

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

# Rutas a los directorios relevantes
REFERENCES_DIR = Path("documentos/Referencias")
PROPOSALS_DIR = Path("documentos/Propuestas")

@tool
def find_similar_proposals(params_str: str) -> str:
    """
    Busca propuestas técnicas similares en el repositorio de propuestas.
    
    Args:
        params_str: String en formato JSON con los siguientes campos:
               - section_name: Nombre de la sección que se va a generar
               - tdr_info: Información extraída del TDR actual
        
    Returns:
        JSON con información sobre propuestas similares encontradas
    """
    # Delega la función al nuevo sistema CRAG
    return crag_find_similar_proposals(params_str)

def extract_keywords(tdr_info: str, section_name: str) -> list:
    """
    Extrae palabras clave relevantes del TDR y el nombre de la sección.
    
    Args:
        tdr_info: Información extraída del TDR
        section_name: Nombre de la sección
        
    Returns:
        Lista de palabras clave
    """
    # Convertir a minúsculas
    tdr_info_lower = tdr_info.lower()
    section_name_lower = section_name.lower()
    
    # Intentar extraer el título del proyecto y otras palabras clave
    keywords = []
    
    # Intentar extraer JSON si es posible
    try:
        info_dict = json.loads(tdr_info)
        
        # Extraer título y palabras clave
        if "titulo_proyecto" in info_dict:
            title_words = info_dict["titulo_proyecto"].lower().split()
            keywords.extend([word for word in title_words if len(word) > 3])
        
        # Extraer algunas palabras clave de requisitos técnicos
        if "requisitos_tecnicos" in info_dict:
            req_text = info_dict["requisitos_tecnicos"].lower()
            # Buscar términos técnicos comunes
            tech_keywords = ["web", "móvil", "cloud", "seguridad", "api", 
                            "inteligencia artificial", "machine learning", "base de datos",
                            "desarrollo", "infraestructura", "red", "hardware", "software"]
            
            for keyword in tech_keywords:
                if keyword in req_text:
                    keywords.append(keyword)
    except:
        # Si no es JSON, extraer palabras clave del texto
        # Extraer palabras clave de secciones específicas
        try:
            # Buscar título del proyecto
            title_match = re.search(r"título del proyecto[:\s]+(.*?)(?:\n|$)", tdr_info_lower)
            if title_match:
                title_words = title_match.group(1).split()
                keywords.extend([word for word in title_words if len(word) > 3])
        except:
            pass
    
    # Añadir palabras clave de la sección
    section_words = section_name_lower.split()
    keywords.extend([word for word in section_words if len(word) > 3])
    
    # Eliminar duplicados y palabras comunes
    common_words = ["para", "como", "esta", "este", "estos", "estas", "que", "los", "las", "del", "con"]
    keywords = [word for word in keywords if word not in common_words]
    
    # Eliminar duplicados
    return list(set(keywords))

def find_reference_excel() -> str:
    """
    Busca el archivo Excel de referencias en la carpeta especificada.
    
    Returns:
        Ruta al archivo Excel o None si no se encuentra
    """
    try:
        # Verificar si el directorio existe
        if not os.path.exists(REFERENCES_DIR):
            logger.warning(f"Directorio de referencias no encontrado: {REFERENCES_DIR}")
            return None
        
        # Buscar archivos Excel
        excel_files = list(REFERENCES_DIR.glob("*.xlsx"))
        if not excel_files:
            excel_files = list(REFERENCES_DIR.glob("*.xls"))
        
        if not excel_files:
            logger.warning(f"No se encontraron archivos Excel en {REFERENCES_DIR}")
            return None
        
        # Devolver el primer archivo Excel encontrado
        return str(excel_files[0])
    except Exception as e:
        logger.error(f"Error al buscar archivo Excel de referencias: {str(e)}", exc_info=True)
        return None

def read_reference_excel(excel_path: str) -> pd.DataFrame:
    """
    Lee el archivo Excel de referencias y extrae la información relevante.
    
    Args:
        excel_path: Ruta al archivo Excel
        
    Returns:
        DataFrame con la información de propuestas anteriores
    """
    try:
        # Leer el Excel - leer específicamente la primera página llamada "PRUEBA 1"
        try:
            # Intenta leer por nombre de hoja
            df = pd.read_excel(excel_path, sheet_name="PRUEBA 1")
            logger.info(f"Lectura exitosa de la hoja 'PRUEBA 1' en {excel_path}")
        except:
            # Si no encuentra por nombre, leer la primera hoja (índice 0)
            df = pd.read_excel(excel_path, sheet_name=0)
            logger.info(f"Lectura exitosa de la primera hoja en {excel_path}")
            
        # Registrar las columnas encontradas para depuración
        logger.info(f"Columnas encontradas en Excel: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        logger.error(f"Error al leer Excel de referencias: {str(e)}", exc_info=True)
        return None

def find_matching_proposals(references: pd.DataFrame, keywords: list) -> list:
    """
    Encuentra propuestas que coincidan con las palabras clave.
    
    Args:
        references: DataFrame con la información de propuestas
        keywords: Lista de palabras clave a buscar
        
    Returns:
        Lista de propuestas coincidentes con puntuación de similitud
    """
    matching_proposals = []
    
    # Verificar las columnas disponibles
    available_columns = references.columns.tolist()
    logger.info(f"Columnas disponibles en el Excel: {available_columns}")
    
    # Buscar la columna que contiene el código del proyecto
    codigo_col = None
    for col in available_columns:
        if "código" in col.lower() or "codigo" in col.lower() or "project" in col.lower():
            codigo_col = col
            break
    
    # Si no encontramos una columna con ese nombre, usar la segunda columna
    # (típicamente, la primera puede ser un índice o numeración y la segunda el código)
    if not codigo_col and len(available_columns) > 1:
        codigo_col = available_columns[1]
        logger.info(f"Usando columna alternativa para códigos: {codigo_col}")
    
    # Si aún no tenemos una columna, usar la primera disponible
    if not codigo_col and available_columns:
        codigo_col = available_columns[0]
        logger.info(f"Usando primera columna disponible: {codigo_col}")
    
    # Si no hay columnas, no podemos continuar
    if not codigo_col:
        logger.warning("No se encontraron columnas en el Excel")
        return []
    
    logger.info(f"Columna seleccionada para códigos de proyecto: {codigo_col}")
    
    # Iterar por las filas del DataFrame
    for index, row in references.iterrows():
        # Obtener el código del proyecto
        project_code = str(row.get(codigo_col, "")).strip()
        
        # Si el código está vacío, continuamos con la siguiente fila
        if not project_code:
            continue
        
        # Calcular una puntuación basada en coincidencias con palabras clave
        score = 0
        project_code_lower = project_code.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in project_code_lower:
                score += 5  # Alta puntuación para coincidencias en el código
        
        # Si no hay palabras clave o son muy genéricas, asignar una puntuación mínima
        if not keywords or score == 0:
            score = 1
        
        # Crear propuesta si hay un código
        matched_prop = {
            "codigo_proyecto": project_code,
            "nombre_proyecto": str(row.get("Nombre del propyecto", "")) if "Nombre del propyecto" in row else "Propuesta " + project_code,
            "ruta_archivo": "",  # No usamos ruta del Excel, buscaremos por código
            "similarity_score": score
        }
        matching_proposals.append(matched_prop)
    
    # Ordenar por puntuación de similitud (mayor a menor)
    matching_proposals.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Limitar a las 5 propuestas más relevantes
    return matching_proposals[:5]

def find_best_column_match(available_columns, candidates):
    """
    Encuentra la mejor coincidencia para una columna entre las candidatas.
    
    Args:
        available_columns: Lista de columnas disponibles
        candidates: Lista de posibles nombres para la columna
        
    Returns:
        Nombre de la columna que mejor coincide o None si no hay coincidencia
    """
    # Primero buscar coincidencia exacta
    for col in available_columns:
        if col.lower() in [c.lower() for c in candidates]:
            return col
    
    # Si no hay coincidencia exacta, buscar coincidencia parcial
    for col in available_columns:
        for candidate in candidates:
            if candidate.lower() in col.lower() or col.lower() in candidate.lower():
                return col
    
    return None

def extract_proposal_content(proposal: dict, section_name: str) -> str:
    """
    Extrae el contenido relevante de una propuesta previa para la sección específica.
    
    Args:
        proposal: Información de la propuesta
        section_name: Nombre de la sección
        
    Returns:
        Contenido extraído para la sección o cadena vacía si no se encuentra
    """
    try:
        # Obtener el código del proyecto
        project_code = str(proposal.get("codigo_proyecto", "")).strip()
        logger.info(f"Buscando contenido para proyecto con código: {project_code}")
        
        # Si el código está vacío, no podemos continuar
        if not project_code or project_code == "UNKNOWN":
            logger.warning("Código de proyecto vacío o desconocido")
            return ""
        
        # Usar únicamente la ruta específica a la carpeta de Propuestas
        base_path = PROPOSALS_DIR  # Esta variable ya está definida como Path("documentos/Propuestas")
        
        if not os.path.exists(base_path):
            logger.warning(f"La carpeta de propuestas no existe: {base_path}")
            return ""
            
        logger.info(f"Buscando en directorio: {base_path}")
        
        # Buscar archivos que contengan el código de proyecto
        potential_files = []
        
        # Búsqueda en la carpeta de propuestas
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            # Solo considerar archivos, no directorios
            if os.path.isfile(item_path):
                # Comprobar si el código está en el nombre del archivo
                if project_code.lower() in item.lower():
                    logger.info(f"Archivo encontrado que coincide con código {project_code}: {item_path}")
                    potential_files.append(item_path)
        
        if not potential_files:
            logger.warning(f"No se encontraron archivos para el código de proyecto: {project_code}")
            return ""
        
        # Ordenar por tipo de archivo (.txt primero)
        def file_priority(file_path):
            if file_path.endswith('.txt'):
                return 0
            elif file_path.endswith('.md'):
                return 1
            elif file_path.endswith('.pdf'):
                return 2
            else:
                return 3
        
        potential_files.sort(key=file_priority)
        
        # Usar el primer archivo encontrado después de ordenar
        file_path = potential_files[0]
        logger.info(f"Usando archivo de mayor prioridad: {file_path}")
        
        # Verificar si es un PDF
        if file_path.lower().endswith('.pdf'):
            try:
                from tools.pdf_tools import extract_text_from_pdf
                content = extract_text_from_pdf(file_path)
                if content.startswith("Error:"):
                    logger.error(f"Error extrayendo texto del PDF: {content}")
                    return ""
                logger.info(f"Texto extraído exitosamente del PDF: {len(content)} caracteres")
            except Exception as e:
                logger.error(f"Error al procesar PDF: {str(e)}")
                return ""
        else:
            # Leer el contenido del archivo para formatos de texto
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                logger.info(f"Archivo leído exitosamente: {len(content)} caracteres")
            except Exception as e:
                logger.error(f"Error al leer archivo: {str(e)}")
                return ""
        
        # Buscar la sección específica utilizando un enfoque más simple
        section_content = extract_section_by_patterns(content, section_name)
        if section_content:
            logger.info(f"Sección '{section_name}' encontrada: {len(section_content)} caracteres")
        else:
            logger.warning(f"Sección '{section_name}' no encontrada en {file_path}")
            
        return section_content
    
    except Exception as e:
        logger.error(f"Error al extraer contenido de propuesta: {str(e)}", exc_info=True)
        return ""

def extract_section_by_patterns(content: str, section_name: str) -> str:
    """
    Extrae una sección específica del contenido utilizando patrones predefinidos.
    
    Args:
        content: Contenido del documento
        section_name: Nombre de la sección a buscar
        
    Returns:
        Contenido de la sección o cadena vacía si no se encuentra
    """
    # Normalizar el texto para búsqueda
    section_name_lower = section_name.lower()
    content_lower = content.lower()
    
    # Lista de patrones para buscar la sección (de más específico a más general)
    patterns = [
        # Patrón de título con número (ej: "2. Objetivos")
        rf'\n\s*\d+\.\s*{re.escape(section_name_lower)}[^\n]*\n(.*?)(?:\n\s*\d+\.\s|\Z)',
        
        # Patrón de título con hashtags (formato markdown)
        rf'\n\s*#+\s*{re.escape(section_name_lower)}[^\n]*\n(.*?)(?:\n\s*#+\s|\Z)',
        
        # Patrón de título en mayúsculas
        rf'\n\s*{re.escape(section_name_lower.upper())}[^\n]*\n(.*?)(?:\n\s*[A-Z][A-Z\s]+|\Z)',
        
        # Patrón de título simple
        rf'\n\s*{re.escape(section_name_lower)}[^\n]*\n(.*?)(?:\n\s*[a-zA-Z]+\s*(?:\:|\.)|\Z)'
    ]
    
    # Probar cada patrón
    for pattern in patterns:
        try:
            match = re.search(pattern, content_lower, re.DOTALL | re.IGNORECASE)
            if match and match.group(1):
                # Encontrar índices en el texto original
                start_idx = content_lower.find(match.group(1))
                if start_idx >= 0:
                    # Extraer el texto del contenido original para preservar mayúsculas/minúsculas
                    section_text = content[start_idx:start_idx + len(match.group(1))].strip()
                    
                    # Limitar longitud si es necesario
                    if len(section_text) > 5000:
                        section_text = section_text[:5000] + "..."
                    
                    return section_text
        except Exception as e:
            logger.warning(f"Error con patrón regex: {str(e)}")
    
    # Si llegamos aquí, no encontramos la sección con los patrones predefinidos
    # Intentar una búsqueda más simple
    try:
        # Encontrar párrafos que contengan el nombre de la sección
        paragraphs = re.split(r'\n\s*\n', content)
        relevant_paragraphs = []
        
        for para in paragraphs:
            para_lower = para.lower()
            if section_name_lower in para_lower:
                relevant_paragraphs.append(para)
        
        if relevant_paragraphs:
            # Unir los párrafos relevantes (máximo 3)
            section_text = "\n\n".join(relevant_paragraphs[:3])
            
            # Limitar longitud
            if len(section_text) > 5000:
                section_text = section_text[:5000] + "..."
                
            return section_text
    except Exception as e:
        logger.warning(f"Error en búsqueda alternativa: {str(e)}")
    
    # No se encontró la sección
    return ""

def get_section_pattern(section_name: str) -> str:
    """
    Genera un patrón regex para encontrar la sección específica.
    
    Args:
        section_name: Nombre de la sección a buscar
        
    Returns:
        Patrón regex para encontrar la sección
    """
    # Normalizar el nombre de la sección
    section_name_lower = section_name.lower()
    
    # Mapear nombre de sección a posibles encabezados
    section_mappings = {
        "introduccion": ["introducción", "introduccion", "1\\.", "i\\.", "1 ", "1\\s+introducción"],
        "objetivos": ["objetivos", "2\\.", "ii\\.", "2 ", "2\\s+objetivos"],
        "alcance": ["alcance", "3\\.", "iii\\.", "3 ", "3\\s+alcance", "alcance del trabajo", "alcance del proyecto"],
        "metodologia": ["metodología", "metodologia", "4\\.", "iv\\.", "4 ", "4\\s+metodología", "enfoque metodológico"],
        "plan": ["plan", "cronograma", "5\\.", "v\\.", "5 ", "5\\s+plan", "planificación"],
        "entregables": ["entregables", "6\\.", "vi\\.", "6 ", "6\\s+entregables", "productos a entregar", "deliverables"],
        "recursos": ["recursos", "7\\.", "vii\\.", "7 ", "7\\s+recursos", "equipo de trabajo", "personal asignado"],
        "riesgos": ["riesgos", "8\\.", "viii\\.", "8 ", "8\\s+riesgos", "gestión de riesgos", "análisis de riesgos"],
        "calidad": ["calidad", "9\\.", "ix\\.", "9 ", "9\\s+calidad", "control de calidad", "aseguramiento de calidad"]
    }
    
    # Buscar coincidencias en el mapeo
    header_options = []
    for key, alternatives in section_mappings.items():
        if key in section_name_lower:
            header_options.extend(alternatives)
    
    # Si no hay coincidencias, usar el nombre de la sección
    if not header_options:
        header_options = [re.escape(section_name)]
    
    # Añadir variantes con números
    # Por ejemplo, si buscamos "entregables", también buscar "6. Entregables", "VI. Entregables", etc.
    header_variations = []
    for opt in header_options:
        # Variante original
        header_variations.append(opt)
        
        # Variante con número y punto
        header_variations.append(f"\\d+\\.\\s*{opt}")
        
        # Variante con número romano
        header_variations.append(f"[IVXivx]+\\.\\s*{opt}")
        
        # Variante con título y subtítulo
        header_variations.append(f"{opt}\\s*[:-]")
    
    # Construir el patrón de búsqueda
    header_pattern = "|".join([f"({opt})" for opt in set(header_variations)])
    
    # Patrón para encontrar la sección y su contenido hasta la siguiente sección
    # Buscamos cualquier encabezado que coincida con nuestras opciones (grupo 1)
    # y luego capturamos todo el texto hasta el siguiente encabezado o fin del documento (grupo 2)
    pattern = f"(?:^|\\n)(?:#{{1,3}}\\s*|\\b)({header_pattern})(?:[\\s:]+|\\n)(.*?)(?=\\n(?:#{{1,3}}\\s*|\\b\\d+\\.\\s|\\b[IVXivx]+\\.\\s|\\b(?:{header_pattern}))|$)"
    
    logger.debug(f"Patrón generado para sección '{section_name}': {pattern[:100]}...")
    
    return pattern
