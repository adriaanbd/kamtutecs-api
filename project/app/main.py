from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings, Settings
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

import base64
import numpy as np
import pytesseract
import cv2
from langdetect import detect_langs
from translate import Translator

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def getTranslate(ocr_str, from_lang, to_lang):
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    split_chunks = []

    clean_str = ocr_str.split('\n')

    for i in range(len(clean_str)):
        split_chunks.append( translator.translate(clean_str[i]) )

    translation = " ".join(split_chunks)

    return translation

def getLanguages(img_str):
    possible_lang = detect_langs(img_str)

    return possible_lang

def ocr(img, oem=3,psm=6, lang='en'):

    config = f'--oem {oem} --psm {psm} -l {lang}'

    grey_img = get_grayscale(cv2.imread(img))
    ocr_text = pytesseract.image_to_string(grey_img, config=config)

    return ocr_text

def ocr_handler(event, context):

    # Extract content from json body
    body_image64 = event['image64']
    oem = event["tess-params"]["oem"]
    psm = event["tess-params"]["psm"]
    lang = event["tess-params"]["lang"]

    # Decode & save inp image to /tmp
    with open("/tmp/saved_img.png", "wb") as f:
      f.write(base64.b64decode(body_image64))

    # Ocr
    ocr_text = ocr("/tmp/saved_img.png",oem=oem,psm=psm,lang=lang)
    detect_lang = str(detect_langs(ocr_text)[0])
    translation = getTranslate(ocr_text, "en", "es")

    # Return the result data in json format
    payload = {
      "ocr": ocr_text,
      "language": detect_lang,
      "translation": translation
    }

    return payload

app = FastAPI()
settings = get_settings()
if settings.environment == 'dev':
 origins = [
     'http://localhost',
     'http://localhost:3000',
     'https://locahost:3000',
 ]
 app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=['*'],
     allow_headers=['*'],
 )

def write_image(image64):
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
    text: str

@app.post('/textract')
def textract(image_data: ImageData, response_model=OCRText):
    image64 = image_data.base64
    decoded_data = base64.b64decode(image64)
    np_data = np.frombuffer(decoded_data,np.uint8)
    imgBGR = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    imgRGB = cv2.cvtColor(imgBGR , cv2.COLOR_BGR2RGB)
    grey_img = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    # grey_img = get_grayscale(cv2.imread(img))
    config = f'--oem 3 --psm 6'
    ocr_text = pytesseract.image_to_string(grey_img, config=config)
    print(ocr_text)
    # translation = getTranslate(ocr_text, "en", "es")
    return ocr_text

# if '__name__' == '__main__':
     # uvicorn.run('main:app', host='0.0.0.0', post=8000, debug=True)
