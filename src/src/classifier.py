import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

# Configuración del Modelo (BART Large MNLI es excelente para clasificación Zero-Shot)
MODEL_NAME = "facebook/bart-large-mnli"

def load_model():
    """
    Carga el modelo en la GPU usando cuantización de 4-bits (NF4).
    Esta función está diseñada para correr en la PC de la universidad (Linux/WSL + GPU).
    """
    print(f"⏳ Cargando modelo {MODEL_NAME} en la GPU...")
    
    # 1. Configuración de Cuantización (Para ahorrar VRAM)
    # Esto reduce el modelo para que quepa holgadamente en los 12GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Cargar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Cargar Modelo con configuración de GPU
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config, # ¡Aquí ocurre la magia de QLoRA!
            device_map="auto" # Reparte el modelo automáticamente en la GPU
        )
    except Exception as e:
        print(f"⚠️ Error cargando BitsAndBytes (Normal en PC de casa): {e}")
        print("Cargando en modo CPU (Lento, solo para pruebas)...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 4. Crear Pipeline de Clasificación
    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer
    )
    
    print("✅ Modelo cargado exitosamente.")
    return classifier

# Instancia global (para no recargar el modelo en cada clic)
# Mañana en la universidad, al importar este script, se cargará el modelo.
AI_CLASSIFIER = None

def classify_document(text, candidate_labels):
    """
    Recibe texto y posibles categorías. Devuelve los puntajes.
    """
    global AI_CLASSIFIER
    
    if AI_CLASSIFIER is None:
        # Carga perezosa (Lazy loading)
        AI_CLASSIFIER = load_model()

    # Truncar texto si es muy largo (BART tiene límite de 1024 tokens)
    # Tomamos los primeros 2000 caracteres como aproximación rápida
    text_to_process = text[:2000]

    print("🧠 Procesando con Transformer...")
    result = AI_CLASSIFIER(text_to_process, candidate_labels)
    
    # Formatear resultado para Gradio (Diccionario {Etiqueta: Puntaje})
    output_scores = {}
    for label, score in zip(result['labels'], result['scores']):
        output_scores[label] = score
        
    return output_scores