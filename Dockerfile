
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-docker.txt

COPY src/predict_batch.py src/predict_batch.py
COPY src/forecast_next_week.py src/forecast_next_week.py
COPY src/monitor_forecast.py src/monitor_forecast.py

CMD ["python", "src/predict_batch.py"]
