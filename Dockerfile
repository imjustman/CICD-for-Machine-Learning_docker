FROM python:3.10-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./App ./App
COPY ./Data ./Data
COPY ./Model ./Model
COPY ./Results ./Results
COPY config_prod.yml .

CMD ["uvicorn", "App.app:app", "--host", "0.0.0.0", "--port", "8000"]