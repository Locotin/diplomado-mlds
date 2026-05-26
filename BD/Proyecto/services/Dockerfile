FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

ARG SERVICE_NAME

COPY requirements.base.txt /app/requirements.base.txt
COPY requirements.${SERVICE_NAME}.txt /app/requirements.${SERVICE_NAME}.txt

RUN pip install --no-cache-dir -r /app/requirements.base.txt -r /app/requirements.${SERVICE_NAME}.txt

COPY services/${SERVICE_NAME} /app

EXPOSE 8000

CMD ["python", "-m", "app.main"]
