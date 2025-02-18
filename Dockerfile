FROM tensorflow/tensorflow:latest-gpu
LABEL authors="grego"

WORKDIR /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y libgl1

RUN pip install --no-cache-dir -r requirements.txt
