FROM python:3.7.5-stretch
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home bitbrocker
WORKDIR /home/bitbrocker

COPY . .
