import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

# Modelo BART
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

def load_model():
    print(f"⏳ Cargando modelo {MODEL_NAME} en la GPU...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config, 
            device_map="auto" 
        )
    except Exception as e:
        print(f"⚠️ Error cargando BitsAndBytes: {e}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer
    )
    
    print("✅ Modelo cargado exitosamente.")
    return classifier

AI_CLASSIFIER = None

def classify_document(text, candidate_labels):
    global AI_CLASSIFIER
    if AI_CLASSIFIER is None:
        AI_CLASSIFIER = load_model()

    text_to_process = text[:2000]
    print("🧠 Procesando con Transformer...")
    result = AI_CLASSIFIER(text_to_process, candidate_labels)
    
    output_scores = {}
    for label, score in zip(result['labels'], result['scores']):
        output_scores[label] = score
        
    return output_scores
