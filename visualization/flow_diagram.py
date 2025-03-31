#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización del flujo de trabajo del sistema multiagente.
"""

import logging
import matplotlib.pyplot as plt
import networkx as nx
from core.execution_tracker import get_execution_path

# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")

def generar_diagrama_flujo(output_file="flujo_tdr.png"):
    """
    Genera un diagrama del flujo del agente basado en la ruta de ejecución.
    
    Args:
        output_file: Nombre del archivo de salida para el diagrama
        
    Returns:
        Ruta del archivo generado
    """
    logger.info("Generando diagrama del flujo de trabajo")
    
    # Crear un nuevo grafo para la visualización
    G = nx.DiGraph()
    
    # Añadir nodos principales del agente
    G.add_node("extract_text", pos=(0, 0), node_type="extraction", label="Extracción\nde Texto")
    G.add_node("analyze_tdr", pos=(2, 0), node_type="analysis", label="Análisis\ndel TDR")
    G.add_node("generate_index", pos=(4, 0), node_type="generation", label="Generación\nde Índice")
    G.add_node("generate_sections", pos=(6, 0), node_type="generation", label="Generación\nde Secciones")
    G.add_node("combine_proposal", pos=(8, 0), node_type="integration", label="Integración\nde Propuesta")
    G.add_node("evaluate_proposal", pos=(10, 0), node_type="evaluation", label="Evaluación\nde Propuesta")
    
    # Añadir aristas básicas
    G.add_edge("extract_text", "analyze_tdr", label="Texto\nextraído")
    G.add_edge("analyze_tdr", "generate_index", label="TDR\nanalizado")
    G.add_edge("generate_index", "generate_sections", label="Índice\ngenerado")
    G.add_edge("generate_sections", "generate_sections", label="Siguiente\nsección")
    G.add_edge("generate_sections", "combine_proposal", label="Todas las\nsecciones")
    G.add_edge("combine_proposal", "evaluate_proposal", label="Propuesta\nintegrada")
    
    # Crear figura
    plt.figure(figsize=(16, 8))
    
    # Posiciones de los nodos
    pos = nx.get_node_attributes(G, 'pos')
    
    # Colores para los nodos según su tipo
    node_colors = {
        'extraction': '#66c2a5',    # Verde azulado para extracción
        'analysis': '#fc8d62',      # Naranja para análisis
        'generation': '#8da0cb',    # Azul para generación
        'integration': '#e78ac3',   # Rosa para integración
        'evaluation': '#a6d854'     # Verde para evaluación
    }
    
    # Preparar colores para cada nodo
    color_map = [node_colors[G.nodes[node].get('node_type', 'extraction')] for node in G.nodes()]
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=color_map, alpha=0.8, 
                          node_shape='o', edgecolors='black', linewidths=2)
    
    # Preparar etiquetas de nodos (con formato)
    node_labels = {}
    for node in G.nodes():
        label = G.nodes[node].get('label', node)
        node_labels[node] = label
    
    # Dibujar etiquetas de nodos
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight='bold')
    
    # Dibujar todas las aristas
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2.0, 
                          edge_color='gray', style='solid', 
                          arrows=True, arrowsize=15, 
                          connectionstyle='arc3,rad=0.1')
    
    # Etiquetas de aristas
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if 'label' in data:
            edge_labels[(u, v)] = data['label']
    
    # Dibujar etiquetas de aristas
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, 
                               font_weight='bold', label_pos=0.5, rotate=False)
    
    # Añadir título
    plt.title("Flujo de Trabajo del Sistema Multiagente para Análisis de TDR", fontsize=16, fontweight='bold')
    
    # Quitar ejes
    plt.axis('off')
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Diagrama guardado como {output_file}")
    plt.close()
    
    return output_file

def generar_diagrama_ejecucion(output_file="ejecucion_tdr.png"):
    """
    Genera un diagrama de la ejecución real del agente basado en el historial de ejecución.
    
    Args:
        output_file: Nombre del archivo de salida para el diagrama
        
    Returns:
        Ruta del archivo generado
    """
    logger.info("Generando diagrama de la ejecución real")
    
    # Obtener el historial de ejecución
    execution_path = get_execution_path()
    
    if not execution_path:
        logger.warning("No hay historial de ejecución para generar el diagrama")
        return None
    
    # Implementación del diagrama de ejecución basado en el historial
    # Esta es una versión simplificada, puede extenderse según sea necesario
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Crear lista de nodos y pasos
    nodes = []
    descriptions = []
    
    for step in execution_path:
        node = step.get("node")
        desc = step.get("description")
        if node and desc:
            nodes.append(node)
            descriptions.append(desc)
    
    # Crear gráfico simple
    plt.plot(range(len(nodes)), [0] * len(nodes), 'ko-')
    
    # Añadir etiquetas
    for i, (node, desc) in enumerate(zip(nodes, descriptions)):
        plt.annotate(
            f"{node}\n{desc[:30]}{'...' if len(desc) > 30 else ''}",
            (i, 0),
            xytext=(0, 10 if i % 2 == 0 else -30),
            textcoords="offset points",
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.title("Historial de Ejecución del Sistema Multiagente")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Diagrama de ejecución guardado como {output_file}")
    plt.close()
    
    return output_file
