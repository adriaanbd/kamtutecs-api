# pull official base image
FROM ubuntu:latest

MAINTAINER Adriaan Beiertz "adriaanbd@gmail.com"

# set working directory
WORKDIR /usr/src/app

# set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# install system dependencies
RUN apt-get update -y && apt dist-upgrade -y\
  && apt-get -y install apt-utils \
  && apt-get -y install netcat gcc \
  && apt-get install -y python3-pip python3-dev build-essential \
  && apt update -y && apt -y dist-upgrade \
  && apt install -y libsm6 libxext6 \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install tesseract-ocr-spa \
  && apt-get -y install tesseract-ocr-por \
  && apt-get -y install tesseract-ocr-ita \
  && apt-get -y install libtesseract-dev \
  && apt-get -y install libleptonica-dev \
  && apt-get clean

# install python dependencies
RUN pip3 install --upgrade pip
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download es_core_news_sm \
    && python3 -m spacy download en_core_web_sm \
    && python3 -m spacy download ja_core_news_sm

# add app
COPY . .
