FROM python:3.6.9-slim-stretch
COPY requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR /app
COPY src/ .
RUN mkdir /app/models /app/logs
COPY models/ /app/models
CMD ["gunicorn","--bind","0.0.0.0:7000","wsgi","--timeout=100"]
