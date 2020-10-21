from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings, Settings
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

app = FastAPI()

if get_settings().environment == 'dev':
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


@dataclass
class BoundingBox:
    x: str
    y: str
    w: str
    h: str

class ImageData(BaseModel):
    base64: str
    bbox: BoundingBox

@app.post('/textract')
async def textract(image_data: ImageData):
    return image_data
