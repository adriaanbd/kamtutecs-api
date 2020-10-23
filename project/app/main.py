from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings, Settings
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

import base64
import numpy as np
import pytesseract
import cv2
from googletrans import Translator
import spacy

model_es = spacy.load('es_core_news_sm')
model_en = spacy.load('en_core_web_sm')

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_translate(ocr_str, to_lang):
    translator = Translator()

    clean_str = ''.join(ocr_str.split('\n'))
    translation = translator.translate(clean_str, dest=to_lang)

    return translation.text

def detect_lang(img_str):
    translator = Translator()
    possible_lang = translator.detect(img_str)
    return possible_lang.lang

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

def write_image_to_png(image64):
    f_name = 'saved_img.png'
    with open(f"./{f_name}", "wb") as f:
        f.write(base64.b64decode(image64))

@dataclass
class BoundingBox:
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

@app.post('/textract')
def textract(image_data: ImageData):
    image64 = image_data.base64
    decoded_data = base64.b64decode(image64)
    np_data = np.frombuffer(decoded_data,np.uint8)
    imgBGR = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    x, y, w, h = image_data.bbox.x, image_data.bbox.y, image_data.bbox.w, image_data.bbox.h
    x, y, w, h = round(float(x)), round(float(y)), round(float(w)), round(float(h))
    resized_img = cv2.resize(imgBGR, (500, 500))
    cropped_img = resized_img[y: y + h, x: x + w]
    grey_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    config = f'--oem 3 --psm 6'
    ocr_text = pytesseract.image_to_string(grey_img, config=config)
    language = detect_lang(ocr_text)
    translation = get_translate(ocr_text, "es")

    nlp_es = model_es(translation)
    nlp_resp = []
    for word in nlp_es:
        nlp_resp.append((word.text, word.pos_))

    return {
        "ocr_text": ocr_text,
        "input_text": language,
        "translation": translation,
        "bbox": image_data.bbox,
        "nlp": nlp_resp
    }

