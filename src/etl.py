import os
from PIL import Image, ImageOps, ImageEnhance
import PyPDF2
from docx import Document

# Bloque de importación segura
try:
    import pytesseract
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    convert_from_path = None

def preprocess_image(image):
    """
    Mejora la imagen para ayudar al OCR:
    1. Convierte a escala de grises.
    2. Aumenta el contraste.
    3. Escala la imagen si es pequeña (Zoom x2).
    """
    # 1. Escala de grises
    image = image.convert('L')
    
    # 2. Aumentar contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0) # Doble contraste
    
    # 3. Escalado (Si es pequeña, la agrandamos)
    width, height = image.size
    if width < 1000:
        new_size = (width * 2, height * 2)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    return image

def extract_text_from_pdf(file_path):
    text = ""
    try:
        # Intento nativo
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text()
                if txt: text += txt + "\n"
        
        # Si falla (PDF Escaneado), usar OCR con pre-procesamiento
        if len(text.strip()) < 50:
            if TESSERACT_AVAILABLE:
                print(f"📄 PDF escaneado detectado: {file_path}. Usando OCR...")
                images = convert_from_path(file_path)
                ocr_text = ""
                for img in images:
                    # ¡Pre-procesamos cada página!
                    img = preprocess_image(img)
                    ocr_text += pytesseract.image_to_string(img, lang='spa') + "\n"
                return ocr_text
            else:
                return "[INFO] PDF Escaneado detectado. Instala poppler-utils y pdf2image."
        return text
    except Exception as e:
        return f"Error leyendo PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error leyendo Word: {e}"

def extract_text_from_image(file_path):
    if not TESSERACT_AVAILABLE:
        return "[ERROR] Tesseract no instalado."
    
    try:
        # Cargar imagen
        image = Image.open(file_path)
        
        # --- MEJORA: Pre-procesamiento ---
        print(f"🖼️ Mejorando calidad de imagen: {file_path}")
        image = preprocess_image(image)
        # ---------------------------------
        
        text = pytesseract.image_to_string(image, lang='spa')
        
        # Validación de "Basura"
        if len(text.strip()) < 5:
            return "[ADVERTENCIA] No se pudo detectar texto claro en la imagen."
            
        return text
    except Exception as e:
        return f"Error de OCR: {e}"

def process_document(file_obj):
    if hasattr(file_obj, 'name'):
        file_path = file_obj.name
    else:
        file_path = file_obj

    ext = os.path.splitext(file_path)[1].lower()
    print(f"Procesando archivo: {file_path}")

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    else:
        return "⚠️ Formato no soportado. Por favor sube PDF, Word o Imagen."
