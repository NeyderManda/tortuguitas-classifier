import gradio as gr
from etl import process_document
import time

# --- CONFIGURACIÓN ---
# Cambia esto a TRUE mañana en la universidad
USAR_GPU_REAL = False 

if USAR_GPU_REAL:
    from classifier import classify_document
else:
    print("⚠️ Modo Simulación Activo (Sin GPU)")

# Definimos las categorías del proyecto "Tortuguitas"
CATEGORIAS = ["Política", "Tecnología", "Deporte", "Entretenimiento"]

def pipeline_principal(file):
    if file is None:
        return "Sube un archivo.", None

    # 1. ETL (Siempre real)
    print("--- 1. Extracción ---")
    texto = process_document(file)
    
    # 2. Clasificación
    print("--- 2. Clasificación ---")
    
    if USAR_GPU_REAL:
        # Llamada al cerebro real (Mañana)
        try:
            scores = classify_document(texto, CATEGORIAS)
            mensaje_extra = "✅ Procesado con BART (GPU)"
        except Exception as e:
            scores = {"Error": 0.0}
            mensaje_extra = f"❌ Error en GPU: {e}"
    else:
        # Simulación (Hoy en casa)
        time.sleep(1) 
        # Lógica tonta basada en palabras clave para testear la UI
        txt_low = texto.lower()
        if "fútbol" in txt_low: scores = {"Deporte": 0.9, "Otros": 0.1}
        elif "ley" in txt_low: scores = {"Política": 0.8, "Otros": 0.2}
        else: scores = {"Tecnología": 0.4, "Entretenimiento": 0.4, "Otros": 0.2}
        mensaje_extra = "⚠️ Modo Simulado (Activa USAR_GPU_REAL en el código)"

    return texto[0:1000] + f"\n\n[...]\n\n{mensaje_extra}", scores

# Interfaz Gráfica
demo = gr.Interface(
    fn=pipeline_principal,
    inputs=gr.File(label="📂 Documento (PDF, DOCX, JPG)", file_count="single"),
    outputs=[
        gr.Textbox(label="🔍 Texto Detectado"),
        gr.Label(label="📊 Clasificación IA", num_top_classes=4)
    ],
    title="🐢 Tortuguitas AI: Clasificador Multimodal",
    description="""
    **Instrucciones:**
    1. Sube un documento (Noticia).
    2. El sistema extraerá el texto (OCR si es necesario).
    3. El modelo BART-Large clasificará el contenido.
    """,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()