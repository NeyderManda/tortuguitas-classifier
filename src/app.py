import gradio as gr
from etl import process_document
import time
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
USAR_GPU_REAL = True 

if USAR_GPU_REAL:
    try:
        from classifier import classify_document
        print("✅ Modo GPU: Clasificador cargado.")
    except ImportError:
        print("⚠️ Advertencia: No se encontró classifier.py, usando modo simulado.")
        USAR_GPU_REAL = False
else:
    print("⚠️ Modo Simulación Activo")

CATEGORIAS = ["Política", "Tecnología", "Deporte", "Entretenimiento"]

# Variable Global para el Conteo Histórico
CONTEO_GLOBAL = {"Política": 0, "Tecnología": 0, "Deporte": 0, "Entretenimiento": 0}

def generar_grafico_confianza(scores):
    """Gráfico de Barras para la noticia actual."""
    if not scores or "Error" in scores: return None
    categorias = list(scores.keys())
    valores = list(scores.values())
    colores = ['#ef4444', '#3b82f6', '#22c55e', '#a855f7']
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(categorias, valores, color=colores[:len(categorias)])
    ax.set_xlim(0, 1.0)
    ax.set_title('Confianza del Modelo (Noticia Actual)')
    plt.tight_layout()
    return fig

def generar_grafico_historial():
    """Gráfico de Pastel (Pie Chart) para el acumulado."""
    etiquetas = []
    valores = []
    colores_map = {'Política': '#ef4444', 'Tecnología': '#3b82f6', 'Deporte': '#22c55e', 'Entretenimiento': '#a855f7'}
    colores = []

    for cat, val in CONTEO_GLOBAL.items():
        if val > 0:
            etiquetas.append(f"{cat} ({val})")
            valores.append(val)
            colores.append(colores_map.get(cat, 'gray'))
    
    if not valores: return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie(valores, labels=etiquetas, autopct='%1.1f%%', colors=colores, startangle=90)
    ax.set_title('Historial de Noticias Clasificadas')
    plt.tight_layout()
    return fig

def obtener_icono(categoria):
    iconos = {"Política": "🏛️", "Tecnología": "💻", "Deporte": "⚽", "Entretenimiento": "🎬"}
    return iconos.get(categoria, "❓")

def procesar_y_clasificar(file, historial):
    if file is None:
        return (
            "Sube un archivo.", None, None, None, historial, gr.update(visible=False)
        )

    nombre_archivo = os.path.basename(file.name)
    texto = process_document(file)
    
    if texto.startswith("[ERROR]") or len(texto.strip()) < 10:
        return texto, None, None, None, historial, gr.update(value="<div style='background:red; color:white; padding:10px;'>Error</div>")

    # Clasificación
    if USAR_GPU_REAL:
        try:
            scores = classify_document(texto, CATEGORIAS)
        except Exception as e:
            scores = {"Error": 0.0}
    else:
        time.sleep(1) 
        txt_low = texto.lower()
        if "fútbol" in txt_low: scores = {"Deporte": 0.9}
        elif "ley" in txt_low: scores = {"Política": 0.8}
        else: scores = {"Tecnología": 0.5}

    # Actualizar Conteo Global
    ganador = max(scores, key=scores.get)
    if ganador in CONTEO_GLOBAL:
        CONTEO_GLOBAL[ganador] += 1

    # Gráficos
    grafico_barras = generar_grafico_confianza(scores)
    grafico_pastel = generar_grafico_historial()

    # Banner
    etiqueta_ganadora = obtener_icono(ganador) + " " + ganador
    colores_bg = {"Política": "#ef4444", "Tecnología": "#3b82f6", "Deporte": "#22c55e", "Entretenimiento": "#a855f7"}
    color_fondo = colores_bg.get(ganador, "gray")
    
    html_banner = f"""
    <div style="background-color: {color_fondo}; color: white; padding: 15px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold;">
        🏆 {etiqueta_ganadora}
    </div>
    """

    # Historial
    nuevo_registro = [nombre_archivo, etiqueta_ganadora, f"{scores[ganador]:.1%}"]
    historial.insert(0, nuevo_registro) 

    return texto, scores, grafico_barras, grafico_pastel, historial, html_banner

def limpiar_todo():
    """Limpia todo para subir uno nuevo."""
    # Devolvemos None al input de archivo para limpiarlo
    return None, "", None, None, "<div style='padding:10px;'>Esperando...</div>"

# --- INTERFAZ ---
tema = gr.themes.Soft(primary_hue="emerald")

with gr.Blocks(theme=tema, title="Tortuguitas AI") as demo:
    historial_state = gr.State(value=[])

    gr.Markdown("# 🐢 Tortuguitas AI: Clasificador Inteligente")
    
    with gr.Row():
        # IZQUIERDA
        with gr.Column(scale=1):
            archivo_input = gr.File(label="📂 Área de Carga", file_types=[".pdf", ".docx", ".png", ".jpg"])
            
            with gr.Row():
                boton_limpiar = gr.Button("🗑️ Limpiar Input", variant="secondary")
                boton_procesar = gr.Button("🚀 Analizar", variant="primary")
            
            gr.Markdown("### 📊 Estadística Global")
            grafico_pastel_output = gr.Plot(label="Total Clasificado")

        # DERECHA
        with gr.Column(scale=2):
            banner_output = gr.HTML(label="Resultado")
            
            with gr.Row():
                grafico_barras_output = gr.Plot(label="Confianza Actual")
                label_output = gr.Label(label="Top Categorías", num_top_classes=4)
            
            texto_output = gr.Textbox(label="📝 Texto Extraído", lines=10, max_lines=15)
            
            gr.Markdown("### 🕒 Historial Reciente")
            tabla_historial = gr.DataFrame(
                headers=["Archivo", "Categoría", "Confianza"],
                datatype=["str", "str", "str"],
                interactive=False
            )

    # Eventos
    boton_procesar.click(
        fn=procesar_y_clasificar,
        inputs=[archivo_input, historial_state],
        outputs=[texto_output, label_output, grafico_barras_output, grafico_pastel_output, tabla_historial, banner_output]
    )
    
    boton_limpiar.click(
        fn=limpiar_todo,
        inputs=None,
        outputs=[archivo_input, texto_output, label_output, grafico_barras_output, banner_output]
    )

if __name__ == "__main__":
    demo.launch(share=True) 
