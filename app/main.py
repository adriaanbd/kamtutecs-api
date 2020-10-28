from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings, Settings
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import List, Tuple

import base64
import numpy as np
import pytesseract
import cv2
from googletrans import Translator
import spacy

### APP
app = FastAPI()

settings = get_settings()
if settings.environment == 'dev':
 origins = [
     'http://localhost',
     'http://localhost:3000',
     'https://locahost:3000',
     'https://localhost:8000',
     'http://localhost:8000',
 ]
 app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=['*'],
     allow_headers=['*'],
 )

class BoundingBox(BaseModel):
    x: str
    y: str
    w: str
    h: str

class ImageData(BaseModel):
    base64: str
    bbox: BoundingBox

class OCRText(BaseModel):
    original_text: str
    translation: str

def get_translation(ocr_str, to_lang):
    """Traduce el texto utilizando Google Translate"""
    translator = Translator()

    clean_str = ''.join(ocr_str.split('\n'))
    translation = translator.translate(clean_str, dest=to_lang)

    return translation.text

def detect_lang(img_str):
    """Devuelve el lenguage mas factible"""
    translator = Translator()
    possible_lang = translator.detect(img_str)
    return possible_lang.lang

def bbox_values_from_dict(box: BoundingBox) -> List[int]:
    """Extrae y redondea los valores x, y, w, h del BoundingBox"""
    box_dict = box.dict()
    bbox_vals = [ # retorna => [x, y, w, h]
        round(float(val))  # "3.3" o "3" => 3
        for val in box_dict.values()  # "val" en { "key": "val" }
    ]
    return bbox_vals

def b64_to_opencv_img(image_b64: str):
    """
    Devuelve un objeto de imagen OpenCV de
    una imagen representada en cadena base64
    """
    decoded_data = base64.b64decode(image_b64)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image_bgr = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    return image_bgr

def resize_and_crop(img_bgr, box: BoundingBox, dimensions: Tuple[int]=None):
    """
    Reajusta la imagen al tamaño establecido en la interfaz de usuario
    y extrae el subsegmento de la imagen señalado por el usuario
    """
    if dimensions is None:
        dimensions = (500, 500)
    resized_img = cv2.resize(img_bgr, dimensions)
    x, y, w, h = bbox_values_from_dict(box)
    cropped_img = resized_img[y: y + h, x: x + w]
    return cropped_img

def preprocess_img(image_bgr, box, dimensions: Tuple[int]=None):
    """Preprocesa la imagen para mejorar el resultado de tesseract"""
    cropped_image = resize_and_crop(image_bgr, box, dimensions)
    grey_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    return grey_img

def ocr(image, box, oem=3, psm=6) -> str:
    """
    Extrae el texto de la imagen dentro de los limites
    señalados por el usuario a traves del BoundingBox
    """
    preprocessed_img = preprocess_img(image, box)
    config = f'--oem {oem} --psm {psm}'
    ocr_text_raw = pytesseract.image_to_string(preprocessed_img, config=config)
    ocr_text = ' '.join(ocr_text_raw.split('\n'))
    return ocr_text

def translate(text: str, to_lang="es"):
    """Traduce el texto de un idioma al otro"""
    language = detect_lang(text)
    translation = get_translation(text, to_lang)
    return translation

def load_spacy_model(language: str):
    """Carga y devuelve el modelo correspondiente de Spacy"""
    models = {
        'es': 'es_core_news_sm',
        'en': 'en_core_web_sm',
        'ja': 'ja_core_news_sm'
    }
    model_name = models.get(language)
    assert model_name is not None, f'Model for {language} missing'
    return spacy.load(model_name)

def analyze_text(text: str, language="es"):
    """Analisis de texto con NLP"""
    nlp_model = load_spacy_model(language)
    nlp = nlp_model(text)
    nlp_response = [
        (word.text, word.pos_)
        for word in nlp
    ]
    return nlp_response

@app.post('/textract')
async def textract(image_data: ImageData):
    """
    API Endpoint para la extracción de texto via OCR,
    procesamiento y analisis.
    """
    image_b64 = image_data.base64
    image_bgr = b64_to_opencv_img(image_b64)
    ocr_text = ocr(image_bgr, image_data.bbox)
    translation = translate(ocr_text)
    nlp_analysis = analyze_text(translation)
    data = {
        "ocr_text": ocr_text,
        "translation": translation,
        "nlp": nlp_analysis
    }
    return JSONResponse(content=data)
