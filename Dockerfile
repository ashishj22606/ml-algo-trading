FROM python:3.9

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    && apt-get clean

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8080

ENV ENV_FILE .env

CMD ["python", "app.py"]

