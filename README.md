# Kamtutecs API

## Getting Started

### Frontend

```bash
$ git clone https://github.com/adriaanbd/kamtutecs.git
$ cd kamtutecs
$ npm install
$ npm start
```

### Backend

#### Docker

```bash
$ git clone https://github.com/adriaanbd/kamtutecs-api.git
$ cd kamtutecs-api
$ docker-compose build
$ docker-compose up
```

#### Without Docker

```bash
$ apt-get -y install tesseract-ocr tesseract-ocr-spa
$ apt-get -y install libtesseract-dev
$ apt-get -y install libleptonica-dev
$ git clone https://github.com/adriaanbd/kamtutecs-api.git
$ cd kamtutecs-api
$ pip install -r requirements.txt
$ uvicorn app.main:app
```
