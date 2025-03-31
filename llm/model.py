#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración del modelo LLM para el sistema multiagente.
"""

import logging
from langchain_community.llms import Ollama
from config.settings import MODEL_NAME
from config.settings import MODEL_EMDE
from langchain_ollama import OllamaEmbeddings
# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

class LLMSingleton:
    """
    Singleton para gestionar la instancia del modelo de lenguaje.
    Asegura que solo exista una instancia del modelo en toda la aplicación.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Obtiene la instancia única del modelo LLM.
        
        Returns:
            Instancia de Ollama configurada con el modelo especificado
        """
        if cls._instance is None:
            logger.info(f"Inicializando modelo Ollama: {MODEL_NAME}")
            cls._instance = Ollama(model=MODEL_NAME)
        return cls._instance

def get_llm():
    """
    Obtiene la instancia del modelo LLM.
    
    Returns:
        Instancia del modelo LLM
    """
    return LLMSingleton.get_instance()
def get_embeddings():
    
    return OllamaEmbeddings(model=MODEL_EMDE)
    