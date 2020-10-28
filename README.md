# Kamtutecs

## Getting Started

### Instructions

1. Setup Backend
2. Setup Frontend
3. Go to [localhost:3000](http://localhost:3000)
4. Upload image with horizontal text in it
5. Draw a bounding box around the desired text to extract it
6. Submit
7. Open Developer Tools (F12)
8. Look at the Console to see the response

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

### Frontend

```bash
$ git clone https://github.com/adriaanbd/kamtutecs.git
$ cd kamtutecs
$ npm install
$ npm start
```
