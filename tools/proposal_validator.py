#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de propuestas técnicas para asegurar coherencia, formato y calidad.
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool
from llm.model import get_llm
from core.execution_tracker import add_to_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

class ProposalValidator:
    """
    Validador de propuestas técnicas completas.
    """
    
    def __init__(self):
        """Inicializa el validador."""
        self.validation_rules = {
            "structure": [
                {"name": "title", "pattern": r"^#\s+PROPUESTA TÉCNICA", "required": True},
                {"name": "intro", "pattern": r"##\s+\d+\.\s+INTRODUCCIÓN", "required": True},
                {"name": "objectives", "pattern": r"##\s+\d+\.\s+OBJETIVOS", "required": True},
                {"name": "scope", "pattern": r"##\s+\d+\.\s+(ALCANCE|ALCANCE DEL SERVICIO)", "required": True},
                {"name": "methodology", "pattern": r"##\s+\d+\.\s+METODOLOG[IÍ]A", "required": True},
                {"name": "plan", "pattern": r"##\s+\d+\.\s+(PLAN DE TRABAJO|CRONOGRAMA)", "required": True},
                {"name": "deliverables", "pattern": r"##\s+\d+\.\s+ENTREGABLES", "required": True},
                {"name": "resources", "pattern": r"##\s+\d+\.\s+(RECURSOS|PERSONAL)", "required": True},
                {"name": "risks", "pattern": r"##\s+\d+\.\s+(RIESGOS|GESTIÓN DE RIESGOS)", "required": True},
                {"name": "quality", "pattern": r"##\s+\d+\.\s+(CALIDAD|PLAN DE CALIDAD)", "required": True},
                {"name": "standards", "pattern": r"##\s+\d+\.\s+(NORMATIVAS|ESTÁNDARES)", "required": True},
                {"name": "experience", "pattern": r"##\s+\d+\.\s+(EXPERIENCIA|PROYECTOS SIMILARES)", "required": True},
                {"name": "annexes", "pattern": r"##\s+\d+\.\s+(ANEXOS|ANEXOS TÉCNICOS)", "required": False}
            ],
            "content": [
                {"name": "foreign_chars", "pattern": r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', "allowed": False},
                {"name": "think_tags", "pattern": r'<think>.*?</think>', "allowed": False},
                {"name": "empty_sections", "pattern": r'##\s+\d+\.\s+[^\n]+\n\s*\n\s*##', "allowed": False},
                {"name": "code_blocks", "pattern": r'```', "max_occurrences": 5},
                {"name": "tables", "pattern": r'\|[-\s|]+\|', "min_occurrences": 0}
            ],
            "numbering": [
                {"name": "consistent_chapter_numbering", "pattern": r'##\s+(\d+)\.', "must_be_sequential": True},
                {"name": "consistent_subsection_numbering", "pattern": r'###\s+(\d+\.\d+)', "must_be_hierarchical": True}
            ]
        }
    
    def validate_proposal(self, proposal_text: str) -> Dict[str, Any]:
        """
        Valida una propuesta técnica completa.
        
        Args:
            proposal_text: Texto completo de la propuesta
            
        Returns:
            Diccionario con resultados de la validación
        """
        results = {
            "is_valid": True,
            "structure": {"valid": True, "issues": []},
            "content": {"valid": True, "issues": []},
            "numbering": {"valid": True, "issues": []},
            "suggestions": []
        }
        
        # Validar estructura
        for rule in self.validation_rules["structure"]:
            found = re.search(rule["pattern"], proposal_text, re.MULTILINE | re.IGNORECASE)
            if rule["required"] and not found:
                results["is_valid"] = False
                results["structure"]["valid"] = False
                results["structure"]["issues"].append(f"Falta la sección obligatoria: {rule['name']}")
                results["suggestions"].append(f"Añadir la sección: {rule['name']}")
        
        # Validar contenido
        for rule in self.validation_rules["content"]:
            matches = re.findall(rule["pattern"], proposal_text, re.MULTILINE | re.DOTALL)
            
            if "allowed" in rule and rule["allowed"] is False and matches:
                results["is_valid"] = False
                results["content"]["valid"] = False
                results["content"]["issues"].append(f"Contenido no permitido encontrado: {rule['name']}")
                
                if rule["name"] == "foreign_chars":
                    results["suggestions"].append("Eliminar caracteres en idiomas diferentes al español")
                elif rule["name"] == "think_tags":
                    results["suggestions"].append("Eliminar etiquetas <think> y su contenido")
                elif rule["name"] == "empty_sections":
                    results["suggestions"].append("Completar las secciones vacías con contenido relevante")
            
            if "max_occurrences" in rule and len(matches) > rule["max_occurrences"]:
                results["content"]["issues"].append(f"Demasiadas ocurrencias de {rule['name']}: {len(matches)} > {rule['max_occurrences']}")
                results["suggestions"].append(f"Reducir el número de {rule['name']}")
            
            if "min_occurrences" in rule and len(matches) < rule["min_occurrences"]:
                results["content"]["issues"].append(f"Muy pocas ocurrencias de {rule['name']}: {len(matches)} < {rule['min_occurrences']}")
                results["suggestions"].append(f"Aumentar el número de {rule['name']}")
        
        # Validar numeración
        for rule in self.validation_rules["numbering"]:
            matches = re.findall(rule["pattern"], proposal_text, re.MULTILINE)
            
            if "must_be_sequential" in rule and rule["must_be_sequential"]:
                try:
                    numbers = [int(m) for m in matches]
                    expected = list(range(1, len(numbers) + 1))
                    
                    if numbers != expected:
                        results["is_valid"] = False
                        results["numbering"]["valid"] = False
                        results["numbering"]["issues"].append(f"Numeración no secuencial en {rule['name']}: {numbers} != {expected}")
                        results["suggestions"].append("Corregir la numeración de secciones para que sea secuencial")
                except ValueError:
                    pass
            
            if "must_be_hierarchical" in rule and rule["must_be_hierarchical"]:
                if matches:
                    # Verificar que las subsecciones sean jerárquicas (1.1, 1.2, 2.1, etc.)
                    chapter_pattern = re.compile(r'(\d+)\.(\d+)')
                    previous_chapter = None
                    previous_section = None
                    
                    for subsection in matches:
                        match = chapter_pattern.match(subsection)
                        if match:
                            current_chapter = int(match.group(1))
                            current_section = int(match.group(2))
                            
                            if previous_chapter is None:
                                # Primera subsección
                                previous_chapter = current_chapter
                                previous_section = current_section
                            elif current_chapter == previous_chapter:
                                # Misma sección, verificar que la subsección incremente en 1
                                if current_section != previous_section + 1:
                                    results["numbering"]["valid"] = False
                                    results["numbering"]["issues"].append(f"Subsección no secuencial: {previous_chapter}.{previous_section} -> {current_chapter}.{current_section}")
                                previous_section = current_section
                            elif current_chapter > previous_chapter:
                                # Nueva sección, la subsección debe ser 1
                                if current_section != 1:
                                    results["numbering"]["valid"] = False
                                    results["numbering"]["issues"].append(f"Primera subsección no es 1: {current_chapter}.{current_section}")
                                previous_chapter = current_chapter
                                previous_section = current_section
                            else:
                                # Sección anterior, error
                                results["numbering"]["valid"] = False
                                results["numbering"]["issues"].append(f"Retroceso en la numeración de secciones: {previous_chapter}.{previous_section} -> {current_chapter}.{current_section}")
        
        # Actualizar resultado general
        results["is_valid"] = results["structure"]["valid"] and results["content"]["valid"] and results["numbering"]["valid"]
        
        return results
    
    def fix_issues(self, proposal_text: str, validation_results: Dict[str, Any]) -> str:
        """
        Intenta corregir automáticamente los problemas detectados.
        
        Args:
            proposal_text: Texto de la propuesta original
            validation_results: Resultados de la validación
            
        Returns:
            Texto de la propuesta corregido
        """
        corrected_text = proposal_text
        
        # Corregir caracteres extranjeros
        corrected_text = re.sub(r'[^\x00-\x7F\xC0-\xFF\u20AC\xA1-\xBF]+', '', corrected_text)
        
        # Corregir etiquetas <think>
        corrected_text = re.sub(r'<think>.*?</think>', '', corrected_text, flags=re.DOTALL)
        
        # Corregir numeración de secciones principales
        if not validation_results["numbering"]["valid"]:
            section_pattern = r'##\s+\d+\.\s+'
            section_matches = re.findall(section_pattern, corrected_text)
            
            for i, match in enumerate(section_matches):
                new_number = f"## {i+1}. "
                corrected_text = corrected_text.replace(match, new_number)
        
        return corrected_text

@tool
def validate_and_fix_proposal(proposal_text: str) -> str:
    """
    Valida una propuesta técnica y corrige problemas comunes.
    
    Args:
        proposal_text: Texto completo de la propuesta
        
    Returns:
        JSON con resultados de validación y texto corregido
    """
    logger.info("Validando propuesta técnica")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "validate_proposal",
        "Validando formato y coherencia de la propuesta técnica"
    )
    
    try:
        validator = ProposalValidator()
        validation_results = validator.validate_proposal(proposal_text)
        
        # Si hay problemas, intentar corregir
        if not validation_results["is_valid"]:
            logger.info(f"Propuesta con problemas: {validation_results['suggestions']}")
            corrected_text = validator.fix_issues(proposal_text, validation_results)
            
            # Validar de nuevo para verificar mejoras
            new_validation = validator.validate_proposal(corrected_text)
            improvement = len(new_validation["suggestions"]) < len(validation_results["suggestions"])
            
            result = {
                "validation": validation_results,
                "corrected": True,
                "improved": improvement,
                "corrected_text": corrected_text
            }
        else:
            logger.info("Propuesta válida, no requiere correcciones")
            result = {
                "validation": validation_results,
                "corrected": False,
                "improved": False,
                "corrected_text": proposal_text
            }
        
        # Registrar resultado
        add_to_execution_path(
            "validate_proposal_result",
            f"Validación completada: {'válida' if validation_results['is_valid'] else 'inválida'}"
        )
        
        return json.dumps(result)
    except Exception as e:
        error_message = f"Error al validar propuesta: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Registrar error
        add_to_execution_path(
            "validate_proposal_error",
            error_message
        )
        
        return json.dumps({
            "error": error_message,
            "corrected": False,
            "corrected_text": proposal_text
        })

@tool
def improve_proposal_with_llm(proposal_text: str, validation_results: str) -> str:
    """
    Utiliza el LLM para mejorar la propuesta basándose en los resultados de la validación.
    
    Args:
        proposal_text: Texto completo de la propuesta
        validation_results: Resultados de validación en formato JSON
        
    Returns:
        Texto de la propuesta mejorado
    """
    logger.info("Mejorando propuesta con LLM")
    
    # Registrar en el historial de ejecución
    add_to_execution_path(
        "improve_proposal_llm",
        "Aplicando mejoras inteligentes a la propuesta"
    )
    
    try:
        # Parsear los resultados de validación
        results = json.loads(validation_results)
        
        # Si no hay problemas, devolver el texto original
        if results.get("validation", {}).get("is_valid", True):
            logger.info("Propuesta ya es válida, no se aplican mejoras")
            return proposal_text
        
        # Construir un prompt con las sugerencias de mejora
        suggestions = results.get("validation", {}).get("suggestions", [])
        issues = []
        
        if "structure" in results.get("validation", {}):
            issues.extend(results["validation"]["structure"].get("issues", []))
        
        if "content" in results.get("validation", {}):
            issues.extend(results["validation"]["content"].get("issues", []))
        
        if "numbering" in results.get("validation", {}):
            issues.extend(results["validation"]["numbering"].get("issues", []))
        
        prompt = f"""
        Necesito que mejores esta propuesta técnica para resolver los siguientes problemas:
        
        PROBLEMAS DETECTADOS:
        {json.dumps(issues, indent=2)}
        
        SUGERENCIAS:
        {json.dumps(suggestions, indent=2)}
        
        INSTRUCCIONES:
        1. Mantén la estructura general de la propuesta.
        2. Corrige los problemas identificados siguiendo las sugerencias.
        3. Asegúrate de que no haya caracteres en idiomas no latinos (como chino o japonés).
        4. Elimina cualquier etiqueta <think> o </think> y su contenido.
        5. Verifica que la numeración de las secciones sea secuencial (1, 2, 3...).
        6. Completa las secciones vacías o con contenido insuficiente.
        7. Mantén el lenguaje técnico y profesional.
        
        Propuesta original:
        {proposal_text}
        
        Asegúrate de que la propuesta siga el formato del documento PKS-537 RQ-01.
        """
        
        # Llamar al LLM para mejorar la propuesta
        llm = get_llm()
        improved_text = llm.invoke(prompt)
        
        logger.info(f"Propuesta mejorada: {len(improved_text)} caracteres")
        
        # Registrar éxito
        add_to_execution_path(
            "improve_proposal_llm_result",
            "Propuesta mejorada exitosamente"
        )
        
        return improved_text
    except Exception as e:
        error_message = f"Error al mejorar propuesta: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Registrar error
        add_to_execution_path(
            "improve_proposal_llm_error",
            error_message
        )
        
        return proposal_text