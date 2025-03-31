#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot de Telegram para interactuar con el sistema multiagente de análisis de TDR.
"""

import time
import logging
import json
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import re 
from config.settings import TELEGRAM_BOT_TOKEN
from core.state import TDRAgentState
from core.execution_tracker import reset_execution_path, get_execution_path, execution_path
from graph.builder import build_tdr_agent
from visualization.flow_diagram import generar_diagrama_flujo
from tools.evaluation_tools import generate_proposal_summary , filename
from langchain_core.messages import SystemMessage
from config.settings import SYSTEM_PROMPT
from datetime import datetime
import glob
# Configuración del logger
logger = logging.getLogger("TDR_Agente_LangGraph")
def buscar_archivo_mas_cercano():
    # Obtener hora actual
    hora_actual = datetime.now()
    
    # Buscar todos los archivos con el formato específico
    patron_busqueda = os.path.join("propuestas", "Propuesta_TDR_*.txt")
    archivos = glob.glob(patron_busqueda)
    
    mejor_archivo = None
    menor_diferencia = float('inf')
    
    # Patrón para extraer la fecha y hora del nombre del archivo
    patron_fecha = re.compile(r'Propuesta_TDR_(\d{14})\.txt$')
    
    for archivo in archivos:
        # Extraer la fecha del nombre del archivo
        match = patron_fecha.search(archivo)
        if match:
            fecha_str = match.group(1)
            try:
                # Convertir a objeto datetime
                fecha_archivo = datetime.strptime(fecha_str, "%Y%m%d%H%M%S")
                
                # Calcular diferencia en segundos con la hora actual
                diferencia = abs((fecha_archivo - hora_actual).total_seconds())
                
                # Actualizar si encontramos una diferencia menor
                if diferencia < menor_diferencia:
                    menor_diferencia = diferencia
                    mejor_archivo = archivo
            except ValueError:
                continue
    
    return mejor_archivo
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Mensaje de bienvenida del bot.
    
    Args:
        update: Actualización de Telegram
        context: Contexto de Telegram
    """
    keyboard = [
        [InlineKeyboardButton("Información del Sistema", callback_data="info")],
        [InlineKeyboardButton("Requisitos del TDR", callback_data="requirements")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 *Bienvenido al Bot de Análisis de TDR y Generación de Propuestas Técnicas*\n\n"
        "Este bot utiliza un sistema multiagente con LangGraph y Ollama para procesar Términos de Referencia "
        "y generar propuestas técnicas profesionales según los requisitos del documento PKS-537 RQ-01.\n\n"
        "Para comenzar, envía un archivo PDF con el TDR que deseas analizar.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja los callbacks de los botones inline.
    
    Args:
        update: Actualización de Telegram
        context: Contexto de Telegram
    """
    query = update.callback_query
    await query.answer()
    
    if query.data == "info":
        await query.message.reply_text(
            "ℹ️ *Información del Sistema*\n\n"
            "Este sistema multiagente está compuesto por 6 agentes especializados:\n\n"
            "1️⃣ *Agente Extractor*: Extrae texto de documentos PDF\n"
            "2️⃣ *Agente Analista*: Analiza el TDR para extraer información clave\n"
            "3️⃣ *Agente Estructurador*: Crea el índice para la propuesta técnica\n"
            "4️⃣ *Agente Redactor*: Genera el contenido para cada sección\n"
            "5️⃣ *Agente Integrador*: Combina todas las secciones en un documento final\n"
            "6️⃣ *Agente Evaluador*: Valida la calidad de la propuesta\n\n"
            "El modelo utilizado es: `deepseek-r1:1.5b:32b`",
            parse_mode="Markdown"
        )
    elif query.data == "requirements":
        await query.message.reply_text(
            "📋 *Requisitos del Documento PKS-537 RQ-01*\n\n"
            "Las propuestas técnicas deben incluir obligatoriamente:\n\n"
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
            "12. Anexos técnicos",
            parse_mode="Markdown"
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Procesa el archivo PDF recibido y ejecuta el flujo de trabajo.
    
    Args:
        update: Actualización de Telegram
        context: Contexto de Telegram
    """
    # Verificar que sea un PDF
    document = update.message.document
    if document.mime_type != "application/pdf":
        await update.message.reply_text("❌ Por favor, envía un archivo PDF.")
        return
    
    # Informar al usuario que se inicia el procesamiento
    progress_message = await update.message.reply_text(
        "📄 PDF recibido. Iniciando el análisis...\n\n"
        "0% ░░░░░░░░░░░░░░░░░░░░"
    )
    
    try:
        # Limpiar el historial de ejecución
        reset_execution_path()
        
        # Descargar el PDF
        file = await document.get_file()
        file_path = f"tdr_{int(time.time())}.pdf"
        pdf_file = await file.download_to_drive(custom_path=file_path)
        logger.info(f"PDF recibido y guardado en: {file_path}")
        
        # Actualizar mensaje de progreso
        await progress_message.edit_text(
            "📄 PDF recibido y guardado.\n\n"
            "20% ████░░░░░░░░░░░░░░░░"
        )
        
        # Crear el agente
        agent = build_tdr_agent()
        logger.info("Agente TDR construido exitosamente")
        
        # Inicializar el estado con mensajes y la ruta del PDF
        state = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)],
            "next_step": "extract_text",
            "pdf_path": file_path
        }
        
        # Informar al usuario que se está extrayendo el texto
        await progress_message.edit_text(
            "🔍 Extrayendo texto del PDF...\n\n"
            "40% ████████░░░░░░░░░░░░"
        )
        
        # Invocar al agente para comenzar el procesamiento
        result = agent.invoke(state)
        logger.info("Procesamiento del TDR completado")
        
        # Actualizar mensaje de progreso
        await progress_message.edit_text(
            "✅ Procesamiento completado.\n\n"
            "100% ████████████████████"
        )
        print("AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",result)
        
        # Tercero: Mostrar resultados de la evaluación (no depende del archivo)
        if "evaluation" in result:
            try:
                eval_dict = json.loads(result["evaluation"])
                status = eval_dict.get("status", "completado")
                puntuacion = eval_dict.get("puntuacion", "N/A")
                
                # Generar resumen ejecutivo si está disponible
                summary = "No disponible"
                summary = generate_proposal_summary(result["evaluation"])
                if summary.startswith("Error:"):
                    summary = "No se pudo generar el resumen ejecutivo."
                
                # Limpiar el resumen si contiene etiquetas "think"
                if "<think>" in summary:
                    summary = summary.split("</think>", 1)[-1].strip()
                
                # Limitar la longitud del resumen
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                
                # Calcular cumplimiento de requisitos
                cumplimiento = eval_dict.get("cumplimiento_requisitos", {})
                total_reqs = len(cumplimiento)
                cumplidos = sum(1 for val in cumplimiento.values() if val)
                porcentaje = int((cumplidos / total_reqs) * 100) if total_reqs > 0 else 0
                
                # MENSAJE 1: Información básica
                message1 = (
                    f"✅ *Propuesta Técnica Generada*\n\n"
                    f"📊 *Evaluación:*\n"
                    f"• Estado: {status.upper()}\n"
                    f"• Puntuación: {puntuacion}/10\n"
                    f"• Cumplimiento: {porcentaje}% ({cumplidos}/{total_reqs} requisitos)"
                )
                
                await update.message.reply_text(message1, parse_mode="Markdown")
                
                # MENSAJE 2: Fortalezas y debilidades
                fortalezas = "\n".join([f"• {item}" for item in eval_dict.get("fortalezas", ["No especificadas."])[:2]])
                debilidades = "\n".join([f"• {item}" for item in eval_dict.get("debilidades", ["No especificadas."])[:2]])
                
                message2 = (
                    f"💪 *Principales fortalezas:*\n{fortalezas}\n\n"
                    f"🔍 *Áreas de mejora:*\n{debilidades}"
                )
                
                await update.message.reply_text(message2, parse_mode="Markdown")
                
                # MENSAJE 3: Resumen y archivo
                message3 = f"📝 *Resumen ejecutivo:*\n{summary}\n\n"
                
                
                
                
                
                await update.message.reply_text(message3, parse_mode="Markdown")
                
                # MENSAJE 4: Recomendaciones si las hay
                if "recomendaciones" in eval_dict and eval_dict["recomendaciones"]:
                    recomendaciones = "\n".join([f"• {item}" for item in eval_dict["recomendaciones"][:3]])
                    message4 = f"📌 *Recomendaciones:*\n{recomendaciones}"
                    await update.message.reply_text(message4, parse_mode="Markdown")
                
            except json.JSONDecodeError:
                await update.message.reply_text(
                    "✅ Propuesta generada y evaluada.",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                "✅ Procesamiento completado, pero no se generó una evaluación.",
                parse_mode="Markdown"
            )
        
        # Cuarto: Enviar el archivo SOLO si existe
    
        try:
            archivo_cercano = buscar_archivo_mas_cercano()
            if archivo_cercano:
                with open(archivo_cercano, "rb") as f:
                    await update.message.reply_document(
                        document=f, 
                        filename=filename,
                        caption="📄 Propuesta Técnica (Formato TXT)"
                    )
                logger.info(f"Archivo enviado: {filename}")
            else:
                logger.warning(f"Archivo no encontrado: {filename}")
                await update.message.reply_text(
                    "⚠️ El archivo de propuesta existe pero no pudo ser localizado exactamente. "
                    "Verifique manualmente en la carpeta del proyecto.",
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"Error al enviar archivo: {str(e)}")
            await update.message.reply_text(f"⚠️ Error al enviar el archivo: {str(e)[:100]}...")
        
        
        # Quinto: Generar y enviar el diagrama del flujo (independiente del archivo)
        try:
            diagram_path = generar_diagrama_flujo()
            with open(diagram_path, "rb") as f:
                await update.message.reply_photo(
                    photo=f, 
                    caption="📊 Diagrama del flujo de trabajo del sistema multiagente"
                )
            logger.info(f"Diagrama enviado: {diagram_path}")
        except Exception as e:
            logger.error(f"Error al generar o enviar diagrama: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
        await update.message.reply_text(
            f"❌ *Error en el procesamiento*\n\n{str(e)}",
            parse_mode="Markdown"
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Muestra un mensaje de ayuda con instrucciones de uso.
    
    Args:
        update: Actualización de Telegram
        context: Contexto de Telegram
    """
    help_text = (
        "🔍 *Instrucciones de uso*\n\n"
        "*Comandos disponibles:*\n"
        "/start - Iniciar el bot y ver mensaje de bienvenida\n"
        "/help - Mostrar este mensaje de ayuda\n"
        "/status - Ver estado del sistema\n\n"
        "*Cómo usar el bot:*\n"
        "1. Envía un archivo PDF con el Término de Referencia (TDR)\n"
        "2. El sistema procesará automáticamente el documento\n"
        "3. Recibirás la propuesta técnica generada según los requisitos PKS-537 RQ-01\n\n"
        "*Formato de la propuesta:*\n"
        "La propuesta generada incluirá todas las secciones requeridas por PKS-537 RQ-01."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Muestra el estado actual del sistema y estadísticas.
    
    Args:
        update: Actualización de Telegram
        context: Contexto de Telegram
    """
    # Obtener el historial de ejecución para mostrar estadísticas
    execution_path = get_execution_path()
    
    # Contar cuántas veces se ha ejecutado cada nodo
    node_counts = {}
    for step in execution_path:
        node = step.get("node", "unknown")
        node_counts[node] = node_counts.get(node, 0) + 1
    
    # Formatear las estadísticas
    stats = "\n".join([f"• {node}: {count} veces" for node, count in node_counts.items()])
    
    status_text = (
        "🖥️ *Estado del Sistema*\n\n"
        f"• Modelo en uso: `deepseek-r1:1.5b:32b`\n"
        f"• Sistema operativo: Activo\n"
        f"• Pasos de ejecución registrados: {len(execution_path)}\n\n"
        f"*Estadísticas de ejecución:*\n{stats if stats else '• No hay datos de ejecución disponibles'}"
    )
    await update.message.reply_text(status_text, parse_mode="Markdown")

def start_bot():
    """Iniciar el bot de Telegram"""
    logger.info("Iniciando bot de Telegram")
    
    # Crear la aplicación
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Añadir handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    
    # Iniciar polling
    application.run_polling()
    logger.info("Bot iniciado. Esperando mensajes...")