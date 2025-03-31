#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramienta de recuperación aumentada por contexto (CRAG) para propuestas técnicas
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

# Importamos dependencias para embeddings y búsqueda vectorial
import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importación para procesamiento de archivos
from tools.pdf_tools import extract_text_from_pdf
from core.execution_tracker import add_to_execution_path
from llm.model import get_llm, get_embeddings

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

# Define las rutas a la carpeta de propuestas técnicas
PROPOSALS_DIR = Path("documentos/Propuestas")

# Clase para representar las propuestas indexadas
class ProposalEmbedding:
    """Clase para manejar el almacenamiento y recuperación de embeddings de propuestas"""
    
    def __init__(self, proposals_dir: Path = PROPOSALS_DIR, persist_directory: str = "propuestas_db"):
        """
        Inicializa el sistema de embeddings para propuestas técnicas.
        
        Args:
            proposals_dir: Ruta a la carpeta que contiene las propuestas
            persist_directory: Directorio donde se guardarán los vectores
        """
        self.proposals_dir = proposals_dir
        self.persist_directory = persist_directory
        self.embedding_function = get_embeddings()
        
        # Verifica si el directorio existe, si no, créalo
        if not os.path.exists(self.proposals_dir):
            os.makedirs(self.proposals_dir)
            logger.info(f"Directorio creado: {self.proposals_dir}")
        
        # Inicializa la base de datos vectorial (si existe) o crea una nueva
        if os.path.exists(self.persist_directory):
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            logger.info(f"Base de datos vectorial cargada desde {self.persist_directory}")
        else:
            self.vector_db = None
            logger.info("No se encontró base de datos vectorial existente")
    
    def index_proposals(self, force_reindex: bool = False) -> None:
        """
        Indexa todas las propuestas técnicas en la carpeta especificada.
        
        Args:
            force_reindex: Si es True, reindexará las propuestas aunque ya exista la BD
        """
        if self.vector_db is not None and not force_reindex:
            logger.info("Las propuestas ya están indexadas. Use force_reindex=True para reindexar.")
            return
        
        logger.info("Comenzando indexación de propuestas técnicas...")
        
        # Registrar en el historial de ejecución
        add_to_execution_path(
            "index_proposals",
            "Indexando propuestas técnicas para recuperación contextual"
        )
        
        documents = []
        
        try:
            # Buscar todos los archivos en la carpeta de propuestas
            for file_path in tqdm(list(self.proposals_dir.glob("**/*.*"))):
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.md', '.docx']:
                    try:
                        # Leer el contenido del archivo
                        if file_path.suffix.lower() == '.pdf':
                            content = extract_text_from_pdf(str(file_path))
                            if content.startswith("Error:"):
                                logger.warning(f"Error al procesar PDF {file_path}: {content}")
                                continue
                        else:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                        
                        # Chunking (dividir en secciones más pequeñas)
                        chunks = self._chunk_document(content)
                        
                        # Crear documentos para Chroma
                        for i, chunk in enumerate(chunks):
                            if len(chunk.strip()) < 100:  # Ignorar chunks muy pequeños
                                continue
                            
                            # Extraer el nombre del proyecto del nombre del archivo
                            project_name = file_path.stem
                            
                            # Agregar documento
                            documents.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": str(file_path),
                                        "project_name": project_name,
                                        "chunk_id": i,
                                        "file_type": file_path.suffix.lower()
                                    }
                                )
                            )
                        
                        logger.info(f"Procesado: {file_path} - {len(chunks)} chunks generados")
                        
                    except Exception as e:
                        logger.error(f"Error al procesar {file_path}: {str(e)}")
            
            # Crear la base de datos vectorial con los documentos
            if documents:
                logger.info(f"Creando base de datos vectorial con {len(documents)} chunks de documentos")
                self.vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                self.vector_db.persist()
                logger.info(f"Base de datos vectorial creada y persistida en {self.persist_directory}")
            else:
                logger.warning("No se encontraron documentos para indexar")
        
        except Exception as e:
            logger.error(f"Error durante la indexación: {str(e)}", exc_info=True)
            # Registrar error
            add_to_execution_path(
                "index_proposals_error",
                f"Error al indexar propuestas: {str(e)}"
            )
            raise
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Divide un documento en chunks más pequeños para mejor procesamiento
        
        Args:
            text: Texto a dividir
            chunk_size: Tamaño aproximado de cada chunk
            chunk_overlap: Superposición entre chunks
            
        Returns:
            Lista de chunks de texto
        """
        # Implementación básica de chunking - primero intentar dividir por secciones
        # Patrones para detectar títulos y secciones
        section_patterns = [
            r'\n#+\s+(.+?)\n',          # Headers markdown
            r'\n(\d+\.\s+.+?)\n',       # Secciones numeradas
            r'\n([A-Z][A-Z\s]+)\n',     # Títulos en mayúsculas
            r'\n(.*?:)\n'               # Títulos con dos puntos al final
        ]
        
        # Intentar dividir por secciones primero
        chunks = []
        remaining_text = text
        
        for pattern in section_patterns:
            if re.search(pattern, '\n' + remaining_text, re.MULTILINE):
                # Dividir por este patrón de sección
                sections = re.split(pattern, '\n' + remaining_text, flags=re.MULTILINE)
                
                # Procesar secciones
                current_chunk = ""
                for i in range(1, len(sections), 2):
                    title = sections[i].strip()
                    content = sections[i+1] if i+1 < len(sections) else ""
                    
                    # Verificar si el chunk actual más título y contenido excede chunk_size
                    if len(current_chunk) + len(title) + len(content) > chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Agregar título y contenido al chunk actual
                    section_text = title + "\n" + content
                    current_chunk += "\n" + section_text
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Si encontramos secciones, no necesitamos probar más patrones
                if chunks:
                    break
        
        # Si la división por secciones no funcionó, dividir por tamaño
        if not chunks:
            # Dividir por párrafos
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Si el chunk actual más el párrafo excede chunk_size
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Incluir superposición
                    current_chunk = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                
                current_chunk += "\n" + paragraph
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # Si aún no tenemos chunks, dividir arbitrariamente
        if not chunks:
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size].strip())
        
        return [c for c in chunks if c.strip()]
    
    def search_similar_content(self, query: str, section_name: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca contenido similar basado en una consulta y nombre de sección
        
        Args:
            query: Consulta para la búsqueda
            section_name: Nombre de la sección para la que se busca contenido
            k: Número de resultados a devolver
            
        Returns:
            Lista de diccionarios con información de los documentos similares
        """
        if self.vector_db is None:
            logger.warning("Base de datos vectorial no inicializada. Intente indexar primero.")
            return []
        
        # Registrar en el historial de ejecución
        add_to_execution_path(
            "search_similar_content",
            f"Buscando contenido similar para la sección '{section_name}'"
        )
        
        try:
            # Construir una consulta combinada
            combined_query = f"Sección: {section_name}. {query}"
            
            # Realizar la búsqueda
            results = self.vector_db.similarity_search_with_score(
                query=combined_query, 
                k=k*3  # Buscamos más resultados para filtrar después
            )
            
            # Filtrar resultados que contengan el nombre de la sección (opcional)
            section_variants = self._get_section_variants(section_name)
            section_filtered_results = []
            
            for doc, score in results:
                # Comprobar si alguna variante del nombre de la sección está en el contenido
                content = doc.page_content.lower()
                if any(variant in content for variant in section_variants):
                    section_filtered_results.append((doc, score))
            
            # Si no hay resultados después del filtro, usar los originales
            if not section_filtered_results and results:
                section_filtered_results = results
            
            # Limitar a k resultados
            final_results = section_filtered_results[:k]
            
            # Formatear resultados
            formatted_results = []
            for doc, score in final_results:
                result = {
                    "project_name": doc.metadata.get("project_name", "Desconocido"),
                    "project_code": Path(doc.metadata.get("source", "")).stem,
                    "section_content": doc.page_content,
                    "similarity_score": float(score),
                    "source": doc.metadata.get("source", "")
                }
                formatted_results.append(result)
            
            logger.info(f"Búsqueda completada: {len(formatted_results)} resultados para '{section_name}'")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error en la búsqueda: {str(e)}", exc_info=True)
            # Registrar error
            add_to_execution_path(
                "search_similar_content_error",
                f"Error en búsqueda: {str(e)}"
            )
            return []
    
    def _get_section_variants(self, section_name: str) -> List[str]:
        """
        Genera variantes del nombre de la sección para búsqueda flexible
        
        Args:
            section_name: Nombre original de la sección
            
        Returns:
            Lista de variantes del nombre de la sección
        """
        section_name = section_name.lower()
        variants = [section_name]
        
        # Mapeo de nombres de sección a posibles variantes
        section_mappings = {
            "introduccion": ["introducción", "introduccion", "1.", "i.", "1 introducción", "1. introducción"],
            "objetivos": ["objetivos", "2.", "ii.", "2 objetivos", "2. objetivos", "objetivo general", "objetivos específicos"],
            "alcance": ["alcance", "3.", "iii.", "3 alcance", "3. alcance", "alcance del trabajo", "alcance del proyecto"],
            "metodologia": ["metodología", "metodologia", "4.", "iv.", "4 metodología", "enfoque metodológico"],
            "plan": ["plan", "cronograma", "5.", "v.", "5 plan", "planificación", "plan de trabajo"],
            "entregables": ["entregables", "6.", "vi.", "6 entregables", "productos a entregar", "deliverables"],
            "recursos": ["recursos", "7.", "vii.", "7 recursos", "equipo de trabajo", "personal asignado"],
            "riesgos": ["riesgos", "8.", "viii.", "8 riesgos", "gestión de riesgos", "análisis de riesgos"],
            "calidad": ["calidad", "9.", "ix.", "9 calidad", "control de calidad", "aseguramiento de calidad"],
            "normativas": ["normativas", "normas", "10.", "x.", "10 normativas", "estándares", "regulaciones"],
            "experiencia": ["experiencia", "11.", "xi.", "11 experiencia", "proyectos similares", "track record"],
            "anexos": ["anexos", "12.", "xii.", "12 anexos", "apéndices", "documentación adicional"]
        }
        
        # Agregar variantes basadas en el mapeo
        for key, alternatives in section_mappings.items():
            if key in section_name or any(alt in section_name for alt in alternatives):
                variants.extend(alternatives)
        
        # Agregar variantes adicionales (con y sin números)
        clean_name = ''.join([c for c in section_name if not c.isdigit() and c not in ".-:"])
        clean_name = clean_name.strip()
        if clean_name and clean_name not in variants:
            variants.append(clean_name)
        
        return list(set(variants))
# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

class EnhancedProposalRetrieval:
    """
    Versión mejorada del sistema de recuperación de propuestas.
    """
    
    def __init__(self, proposals_dir="documentos/Propuestas", persist_directory="propuestas_db"):
        """Inicializa el sistema mejorado de recuperación."""
        self.embedding_system = ProposalEmbedding(proposals_dir, persist_directory)
        
        # Asegurar que la base de datos esté indexada
        if self.embedding_system.vector_db is None:
            logger.info("Indexando propuestas para el primer uso...")
            self.embedding_system.index_proposals()
    
    def get_section_template(self, section_name: str) -> Dict[str, Any]:
        """
        Obtiene una plantilla estandarizada para una sección específica.
        
        Args:
            section_name: Nombre de la sección (ej. "introduccion", "objetivos")
            
        Returns:
            Diccionario con la plantilla de la sección
        """
        # Mapeo de nombres de sección a plantillas estándar
        templates = {
            "introduccion": {
                "titulo": "1. INTRODUCCIÓN",
                "estructura": [
                    "Contexto del proyecto",
                    "Descripción del problema/necesidad",
                    "Propósito de la propuesta",
                    "Beneficios esperados"
                ],
                "formato": "párrafos",
                "longitud_recomendada": "3-4 párrafos"
            },
            "objetivos": {
                "titulo": "2. OBJETIVOS",
                "estructura": [
                    "2.1 OBJETIVO GENERAL",
                    "2.2 OBJETIVOS ESPECÍFICOS (en viñetas)"
                ],
                "formato": "general en párrafo, específicos en viñetas",
                "longitud_recomendada": "1 párrafo + 3-5 viñetas"
            },
            # Añadir plantillas para otras secciones...
        }
        
        # Normalizar el nombre de la sección para la búsqueda
        normalized_name = section_name.lower().strip()
        
        # Obtener la plantilla o una genérica si no existe específica
        return templates.get(normalized_name, {
            "titulo": f"{section_name.upper()}",
            "estructura": ["Información general", "Detalles específicos"],
            "formato": "párrafos y viñetas según convenga",
            "longitud_recomendada": "2-3 párrafos"
        })
    
    def clean_retrieved_content(self, content: str) -> str:
        """
        Limpia el contenido recuperado para eliminar tags no deseados o texto irrelevante.
        
        Args:
            content: Contenido a limpiar
            
        Returns:
            Contenido limpio
        """
        # Eliminar etiquetas <think> y </think> y su contenido
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Eliminar caracteres no latinos (excepto puntuación común)
        content = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', content)
        
        # Eliminar líneas vacías múltiples
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Normalizar encabezados de sección (### a ##)
        content = re.sub(r'^#{3,}\s+', '## ', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def get_enhanced_context(self, section_name: str, tdr_info: str, num_examples: int = 3) -> Dict[str, Any]:
        """
        Obtiene contexto mejorado para generar una sección específica.
        
        Args:
            section_name: Nombre de la sección
            tdr_info: Información extraída del TDR
            num_examples: Número de ejemplos a obtener
            
        Returns:
            Diccionario con contexto enriquecido para la generación
        """
        # Obtener plantilla de la sección
        template = self.get_section_template(section_name)
        
        # Buscar ejemplos similares
        similar_proposals = self.embedding_system.search_similar_content(
            query=f"Sección {section_name}: {tdr_info[:500]}", 
            section_name=section_name,
            k=num_examples
        )
        
        # Procesar y limpiar ejemplos
        cleaned_examples = []
        for prop in similar_proposals:
            content = self.clean_retrieved_content(prop.get("section_content", ""))
            if content:
                cleaned_examples.append({
                    "project_name": prop.get("project_name", "Proyecto no especificado"),
                    "content": content
                })
        
        # Estructurar el contexto enriquecido
        enhanced_context = {
            "template": template,
            "examples": cleaned_examples,
            "instructions": {
                "format_guidelines": f"Usar el título '{template['titulo']}' y seguir la estructura recomendada",
                "content_guidelines": "Mantener coherencia con secciones anteriores y siguientes",
                "style_guidelines": "Usar lenguaje técnico profesional, evitar repeticiones",
                "common_mistakes": "Evitar cambios de idioma, etiquetas visibles, o estructura incoherente"
            }
        }
        
        return enhanced_context
    
    def create_section_prompt(self, section_name: str, tdr_info: str, previous_sections: List[str] = None) -> str:
        """
        Crea un prompt optimizado para generar una sección específica.
        
        Args:
            section_name: Nombre de la sección
            tdr_info: Información extraída del TDR
            previous_sections: Contenido de secciones anteriores (opcional)
            
        Returns:
            Prompt optimizado para el LLM
        """
        # Obtener contexto enriquecido
        context = self.get_enhanced_context(section_name, tdr_info)
        
        # Construir el prompt
        prompt = f"""
        # Instrucciones para generar la sección '{section_name.upper()}' de la propuesta técnica
        
        ## Información del TDR:
        {tdr_info[:1000]}...
        
        ## Formato y estructura requeridos:
        - Título: {context['template']['titulo']}
        - Estructura recomendada: {', '.join(context['template']['estructura'])}
        - Formato: {context['template']['formato']}
        - Longitud aproximada: {context['template']['longitud_recomendada']}
        
        ## Ejemplos de secciones similares en proyectos previos:
        """
        
        # Añadir ejemplos
        for i, example in enumerate(context['examples']):
            prompt += f"""
        ### Ejemplo {i+1} - Proyecto: {example['project_name']}
        {example['content'][:800]}...
        """
        
        # Añadir contexto de secciones previas si está disponible
        if previous_sections:
            prompt += "\n## Secciones previas de esta propuesta (para mantener coherencia):\n"
            for i, prev_section in enumerate(previous_sections[-2:]):  # Solo las últimas 2 secciones
                prompt += f"### Sección previa {i+1}:\n{prev_section[:500]}...\n\n"
        
        # Añadir instrucciones finales
        prompt += f"""
        ## Instrucciones adicionales:
        1. {context['instructions']['format_guidelines']}
        2. {context['instructions']['content_guidelines']}
        3. {context['instructions']['style_guidelines']}
        4. {context['instructions']['common_mistakes']}
        
        IMPORTANTE: Generar SOLAMENTE el contenido de la sección '{section_name}', sin incluir etiquetas <think> o texto en idiomas diferentes al español. Usar numeración coherente con el resto del documento.
        """
        
        return prompt

# Función para integrar con el sistema existente
def get_enhanced_proposal_context(section_name: str, tdr_info: str, previous_sections: List[str] = None) -> str:
    """
    Versión mejorada de la función de recuperación de contexto para propuestas.
    
    Args:
        section_name: Nombre de la sección
        tdr_info: Información extraída del TDR
        previous_sections: Contenido de secciones anteriores (opcional)
        
    Returns:
        Prompt optimizado para generar la sección
    """
    try:
        retriever = EnhancedProposalRetrieval()
        return retriever.create_section_prompt(section_name, tdr_info, previous_sections)
    except Exception as e:
        logger.error(f"Error al obtener contexto mejorado: {str(e)}", exc_info=True)
        # Fallback a la función original en caso de error
        from tools.crag_tools import get_similar_proposals_context
        return get_similar_proposals_context(section_name, tdr_info)

# Modificar la función de generación de secciones para utilizar el nuevo sistema
def enhance_generate_section_tool():
    """
    Mejora la herramienta de generación de secciones utilizando el contexto enriquecido.
    """
    from tools.generation_tools import generate_section
    original_function = generate_section.func
    
    def enhanced_generate_section(params: str) -> str:
        try:
            # Parsear los parámetros
            params_dict = json.loads(params)
            section_name = params_dict.get("section_name", "")
            tdr_info = params_dict.get("info", "")
            previously_generated = params_dict.get("previous_content", "")
            
            # Dividir el contenido previo en secciones
            previous_sections = []
            if previously_generated:
                # Patrón para encontrar secciones por los encabezados
                sections = re.split(r'##\s+[^\n]+', previously_generated)
                if len(sections) > 1:  # El primero es vacío
                    previous_sections = ["##" + s for s in sections[1:]]
            
            # Utilizar la nueva función de contexto
            enhanced_prompt = get_enhanced_proposal_context(
                section_name, 
                tdr_info, 
                previous_sections
            )
            
            # Reemplazar los parámetros originales con el prompt mejorado
            params_dict["enhanced_context"] = enhanced_prompt
            
            # Llamar a la función original con los parámetros enriquecidos
            return original_function(json.dumps(params_dict))
        except Exception as e:
            logger.error(f"Error en enhanced_generate_section: {str(e)}", exc_info=True)
            # Fallback a la función original
            return original_function(params)
    
    # Reemplazar la función original
    generate_section.func = enhanced_generate_section
    
    return "Herramienta de generación de secciones mejorada con éxito"
# Función para integrar con el sistema de herramientas
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
    logger.info("Buscando propuestas técnicas similares con CRAG")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "find_similar_proposals",
        "Buscando propuestas similares mediante recuperación vectorial"
    )
    
    try:
        # Parsear los parámetros
        params_dict = json.loads(params_str)
        section_name = params_dict.get("section_name", "")
        tdr_info = params_dict.get("tdr_info", "")
        
        # Extraer información relevante del TDR para la consulta
        query = extract_query_from_tdr(tdr_info, section_name)
        
        # Inicializar sistema de embeddings
        proposal_embeddings = ProposalEmbedding()
        
        # Si no existe la base de datos, indexar las propuestas
        if proposal_embeddings.vector_db is None:
            logger.info("Indexando propuestas por primera vez...")
            proposal_embeddings.index_proposals()
        
        # Buscar propuestas similares
        similar_proposals = proposal_embeddings.search_similar_content(
            query=query, 
            section_name=section_name
        )
        
        if not similar_proposals:
            return json.dumps({
                "status": "warning",
                "message": "No se encontraron propuestas similares",
                "proposals": []
            })
        
        return json.dumps({
            "status": "success",
            "message": f"Se encontraron {len(similar_proposals)} propuestas similares",
            "proposals": similar_proposals
        })
        
    except Exception as e:
        error_message = f"Error al buscar propuestas similares: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Registrar error
        add_to_execution_path(
            "find_similar_proposals_error",
            error_message
        )
        
        return json.dumps({
            "status": "error",
            "message": error_message,
            "proposals": []
        })

def extract_query_from_tdr(tdr_info: str, section_name: str) -> str:
    """
    Extrae información relevante del TDR para construir una consulta efectiva
    
    Args:
        tdr_info: Información del TDR
        section_name: Nombre de la sección
        
    Returns:
        Consulta optimizada para la búsqueda
    """
    try:
        # Primero, intentar parsear el tdr_info como JSON
        info_dict = json.loads(tdr_info) if isinstance(tdr_info, str) else tdr_info
        
        # Construir una consulta con palabras clave según la sección
        query_parts = []
        
        # Agregar título del proyecto si existe
        if "titulo_proyecto" in info_dict:
            query_parts.append(f"Proyecto: {info_dict['titulo_proyecto']}")
        
        # Agregar texto específico según la sección
        section_key_mapping = {
            "introduccion": ["titulo_proyecto", "contexto"],
            "objetivos": ["objetivos", "metas"],
            "alcance": ["alcance", "requisitos_tecnicos", "alcance_proyecto"],
            "metodologia": ["metodologia", "enfoque", "implementacion"],
            "plan_trabajo": ["cronograma", "plazos", "actividades"],
            "entregables": ["entregables", "productos"],
            "recursos": ["recursos", "equipo", "materiales"],
            "riesgos": ["riesgos", "problemas", "contingencias"],
            "calidad": ["calidad", "estandares"],
            "normativas": ["normativas", "regulaciones", "leyes"],
            "experiencia": ["experiencia", "proyectos_similares"],
            "anexos": ["informacion_adicional", "documentos_adjuntos"]
        }
        
        # Buscar la mejor coincidencia para la sección
        best_match = None
        for key in section_key_mapping:
            if key in section_name.lower():
                best_match = key
                break
        
        if best_match:
            for field in section_key_mapping[best_match]:
                if field in info_dict and info_dict[field]:
                    query_parts.append(f"{field}: {info_dict[field]}")
        
        # Si hay pocos elementos, agregar información general
        if len(query_parts) < 2:
            for key, value in info_dict.items():
                if key not in ["titulo_proyecto"] and isinstance(value, str) and len(value) < 200:
                    query_parts.append(f"{key}: {value}")
        
        return " ".join(query_parts[:5])  # Limitar a 5 partes para evitar query demasiado larga
        
    except (json.JSONDecodeError, TypeError):
        # Si falla el parsing JSON, extraer información mediante técnicas alternativas
        # Primero intentar extraer título y palabras clave
        query_parts = []
        
        # Buscar título del proyecto
        title_match = re.search(r"título del proyecto[:\s]+(.*?)(?:\n|$)", tdr_info, re.IGNORECASE)
        if title_match:
            query_parts.append(f"Proyecto: {title_match.group(1).strip()}")
        
        # Intentar encontrar texto relacionado con la sección
        section_patterns = {
            "introduccion": [r"contexto[:\s]+(.*?)(?:\n\n|$)", r"introducción[:\s]+(.*?)(?:\n\n|$)"],
            "objetivos": [r"objetivos?[:\s]+(.*?)(?:\n\n|$)", r"metas?[:\s]+(.*?)(?:\n\n|$)"],
            "alcance": [r"alcance[:\s]+(.*?)(?:\n\n|$)", r"requisitos[:\s]+(.*?)(?:\n\n|$)"],
            "metodologia": [r"metodolog[ií]a[:\s]+(.*?)(?:\n\n|$)", r"enfoque[:\s]+(.*?)(?:\n\n|$)"],
            "plan_trabajo": [r"cronograma[:\s]+(.*?)(?:\n\n|$)", r"plazos?[:\s]+(.*?)(?:\n\n|$)"],
            "entregables": [r"entregables?[:\s]+(.*?)(?:\n\n|$)", r"productos?[:\s]+(.*?)(?:\n\n|$)"],
            "recursos": [r"recursos?[:\s]+(.*?)(?:\n\n|$)", r"equipo[:\s]+(.*?)(?:\n\n|$)"],
            "riesgos": [r"riesgos?[:\s]+(.*?)(?:\n\n|$)", r"contingencias?[:\s]+(.*?)(?:\n\n|$)"],
            "calidad": [r"calidad[:\s]+(.*?)(?:\n\n|$)", r"est[áa]ndares?[:\s]+(.*?)(?:\n\n|$)"],
            "normativas": [r"normativas?[:\s]+(.*?)(?:\n\n|$)", r"regulaciones?[:\s]+(.*?)(?:\n\n|$)"],
            "experiencia": [r"experiencia[:\s]+(.*?)(?:\n\n|$)", r"proyectos similares[:\s]+(.*?)(?:\n\n|$)"],
        }
        
        # Encontrar el mejor patrón para la sección
        best_match = None
        for key in section_patterns:
            if key in section_name.lower():
                best_match = key
                break
        
        if best_match:
            for pattern in section_patterns[best_match]:
                match = re.search(pattern, tdr_info, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # Limitar longitud
                    if len(content) > 200:
                        content = content[:200] + "..."
                    query_parts.append(content)
        
        # Si no hay suficientes partes, agregar palabras clave del TDR
        if len(query_parts) < 2:
            # Extraer oraciones importantes (primeras 2-3 oraciones)
            sentences = re.split(r'[.!?]+', tdr_info)
            key_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
            query_parts.extend(key_sentences)
        
        return " ".join(query_parts[:3])  # Limitar a 3 partes

def reindex_proposals() -> str:
    """
    Fuerza la reindexación de todas las propuestas técnicas
    
    Returns:
        Mensaje con el resultado de la operación
    """
    try:
        logger.info("Iniciando reindexación forzada de propuestas")
        
        # Registrar en el historial de ejecución
        add_to_execution_path(
            "reindex_proposals",
            "Forzando reindexación de propuestas técnicas"
        )
        
        # Inicializar y reindexar
        proposal_embeddings = ProposalEmbedding()
        proposal_embeddings.index_proposals(force_reindex=True)
        
        return json.dumps({
            "status": "success",
            "message": "Propuestas reindexadas exitosamente"
        })
    except Exception as e:
        error_message = f"Error al reindexar propuestas: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        return json.dumps({
            "status": "error",
            "message": error_message
        })

# Reemplazar la implementación existente en tools/reference_tools.py con esta nueva implementación
# Integración con el sistema existente
def get_similar_proposals_context(section_name: str, tdr_info: str) -> str:
    """
    Obtiene contexto de propuestas similares para la sección actual.
    
    Args:
        section_name: Nombre de la sección
        tdr_info: Información extraída del TDR
        
    Returns:
        Texto con contexto de propuestas similares o cadena vacía si no hay
    """
    try:
        # Preparar parámetros para buscar propuestas similares
        search_params = json.dumps({
            "section_name": section_name,
            "tdr_info": tdr_info
        })
        
        # Buscar propuestas similares
        similar_proposals_json = find_similar_proposals(search_params)
        
        try:
            similar_proposals = json.loads(similar_proposals_json)
            
            if similar_proposals.get("status") != "success" or not similar_proposals.get("proposals"):
                logger.info(f"No se encontraron propuestas similares para la sección: {section_name}")
                return ""
            
            # Construir el contexto a partir de las propuestas encontradas
            context_parts = []
            for i, prop in enumerate(similar_proposals.get("proposals", [])):
                if prop.get("section_content"):
                    context_parts.append(
                        f"### Ejemplo {i+1} - Proyecto: {prop.get('project_name', 'N/A')} (Código: {prop.get('project_code', 'N/A')})\n"
                        f"{prop.get('section_content', '')}\n"
                    )
            
            # Limitar el tamaño total del contexto
            context = "\n".join(context_parts)
            if len(context) > 7000:  # Limitar a 7000 caracteres para evitar problemas con el tamaño del prompt
                context = context[:7000] + "...\n[Contexto truncado por tamaño]"
            
            return context
        except json.JSONDecodeError:
            logger.warning(f"Error al decodificar respuesta JSON: {similar_proposals_json[:200]}...")
            return ""  # Retornar cadena vacía en caso de error
            
    except Exception as e:
        logger.error(f"Error al obtener contexto de propuestas similares: {str(e)}", exc_info=True)
        return ""  # Retornar cadena vacía en caso de error

# Esta función se puede llamar una vez al inicio para indexar todas las propuestas
def initialize_proposal_database():
    """
    Inicializa la base de datos de propuestas si no existe
    
    Returns:
        Estado de la inicialización
    """
    try:
        proposal_embeddings = ProposalEmbedding()
        
        # Si no existe la base de datos, indexar
        if proposal_embeddings.vector_db is None:
            logger.info("Inicializando base de datos de propuestas por primera vez...")
            proposal_embeddings.index_proposals()
            return "Base de datos de propuestas inicializada correctamente"
        else:
            return "Base de datos de propuestas ya existe"
    except Exception as e:
        error_message = f"Error al inicializar base de datos de propuestas: {str(e)}"
        logger.error(error_message, exc_info=True)
        return error_message