#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuraciones globales para el sistema TDR Agente Multiagente.
"""

import os
import logging

# Configuración de Telegram
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Configuración del modelo
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:14b")
MODEL_EMDE = os.getenv("MODEL_EMDE")
# Sistema de instrucciones para el agente
SYSTEM_PROMPT = """Eres un sistema multi-agente especializado en analizar Términos de Referencia (TDR) y generar propuestas técnicas profesionales.

El sistema está compuesto por varios agentes especializados que trabajan en conjunto:

1. Agente Extractor: Extrae texto de documentos PDF
2. Agente Analista: Analiza el TDR para extraer información clave como alcance, entregables, etc.
3. Agente Estructurador: Crea un índice para la propuesta técnica basado en el análisis
4. Agente Redactor: Genera contenido para cada sección de la propuesta
5. Agente Integrador: Combina todas las secciones en un documento cohesivo
6. Agente Evaluador: Verifica la calidad y adecuación de la propuesta al TDR

Cada agente puede acceder a herramientas especializadas para cumplir su función. El resultado final será una propuesta técnica completa que responde a los requisitos del TDR recibido.

Actúa de manera profesional, detallada y precisa en todas las etapas del proceso.
"""

# Configuración de logging
def configure_logging():
    """Configura y devuelve el logger para la aplicación"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("TDR_Agente_LangGraph")
    logger.info("Sistema de logging configurado")
    return logger
