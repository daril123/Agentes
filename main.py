#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDR Agente Multiagente con LangGraph y Ollama
=============================================

Este sistema implementa un agente conversacional multiagente para procesar 
Términos de Referencia (TDR) y generar propuestas técnicas profesionales.

Utiliza LangGraph para definir el flujo de trabajo entre diferentes agentes especializados
y Ollama con el modelo deepseek-r1:1.5b:32b para las tareas de procesamiento de lenguaje natural.

El sistema se integra con Telegram para permitir a los usuarios enviar archivos PDF
y recibir propuestas técnicas generadas automáticamente.
"""

import logging
import json
from config.settings import configure_logging
from telegram_bot.bot import start_bot

from tools.crag_tools import initialize_proposal_database
# Configurar logging
logger = configure_logging()

def initialize_environment():
    """Inicializa el entorno del proyecto"""
    logger.info("Inicializando entorno del proyecto")
    
    # Crear estructura de directorios
    
    logger.info("Inicializando base de datos de propuestas CRAG...")
    crag_init_result = initialize_proposal_database()
    logger.info(f"CRAG: {crag_init_result}")
    # Verificar Excel de referencias
    
    
def main():
    """Función principal para iniciar el sistema"""
    try:
        logger.info("Iniciando el sistema TDR Agente Multiagente")
        
        # Inicializar entorno
        initialize_environment()
        
        # Iniciar el bot de Telegram
        start_bot()
    except Exception as e:
        logger.error(f"Error al iniciar el sistema: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

