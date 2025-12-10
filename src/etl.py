import os
from PIL import Image
import PyPDF2
from docx import Document

# Bloque de importación segura:
# Si no tienes Tesseract instalado hoy, el programa no fallará, solo te avisará.
try:
    import pytesseract
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    convert_from_path = None

def extract_text_from_pdf(file_path):
    """Extrae texto de un PDF. Si es escaneado, pide ayuda al OCR."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text()
                if txt: text += txt + "\n"
        
        # Validación: Si hay muy poco texto, probablemente sea una imagen pegada en el PDF
        if len(text) < 50:
            return "[INFO] Poco texto detectado (PDF Escaneado). Se requeriría OCR (Tesseract) aquí."
        return text
    except Exception as e:
        return f"Error leyendo PDF: {e}"

def extract_text_from_docx(file_path):
    """Extrae texto de un archivo Word."""
    try:
        doc = Document(file_path)
        full_text = []
        for p in doc.paragraphs:
            full_text.append(p.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error leyendo Word: {e}"

def extract_text_from_image(file_path):
    """Intenta leer texto de una imagen."""
    if not TESSERACT_AVAILABLE:
        return "[INFO] Para leer imágenes necesitas instalar Tesseract-OCR. Mañana en la PC de la universidad funcionará directo."
    
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image, lang='spa')
    except Exception as e:
        return f"Error de OCR: {e}"

def process_document(file_obj):
    """
    Función Principal (El 'Gerente').
    Recibe un archivo, mira su extensión y decide a quién llamar.
    """
    # Manejo de diferencias entre Gradio y rutas locales
    if hasattr(file_obj, 'name'):
        file_path = file_obj.name
    else:
        file_path = file_obj

    # Obtener la extensión del archivo (ej: .pdf, .docx)
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