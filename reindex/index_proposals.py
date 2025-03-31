#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para indexar propuestas técnicas en la base de datos vectorial
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger("crag_indexer")

def setup_project_path():
    """Añade la ruta del proyecto al sys.path para poder importar módulos"""
    # Obtener la ruta actual y subir un nivel para llegar a la raíz del proyecto
    current_path = Path(__file__).resolve().parent
    project_path = current_path.parent
    sys.path.append(str(project_path))

def main():
    """Función principal para indexar propuestas"""
    parser = argparse.ArgumentParser(description='Indexar propuestas técnicas para CRAG')
    parser.add_argument('--force', action='store_true', help='Forzar reindexación completa')
    parser.add_argument('--directory', type=str, help='Directorio de propuestas (opcional)')
    args = parser.parse_args()
    
    # Añadir ruta del proyecto al sys.path
    setup_project_path()
    
    try:
        # Importar módulos del proyecto
        from tools.crag_tools import ProposalEmbedding, reindex_proposals
        
        # Si se proporcionó un directorio personalizado, modificar la variable global
        if args.directory:
            directory_path = Path(args.directory)
            if not directory_path.exists():
                logger.error(f"El directorio {args.directory} no existe")
                return 1
                
            # Actualizar la ruta del directorio
            logger.info(f"Usando directorio personalizado: {args.directory}")
            ProposalEmbedding.PROPOSALS_DIR = directory_path
        
        # Reindexar las propuestas
        if args.force:
            logger.info("Iniciando reindexación forzada de propuestas")
            result = reindex_proposals()
            logger.info(f"Resultado: {result}")
        else:
            logger.info("Inicializando base de datos de propuestas")
            proposal_embeddings = ProposalEmbedding()
            proposal_embeddings.index_proposals()
            logger.info("Indexación completada")
            
        return 0
            
    except Exception as e:
        logger.error(f"Error durante la indexación: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())