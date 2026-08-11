FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# Persist the SQLite DB (keys, spend ledger, cache) outside the container layer.
VOLUME ["/app/data"]
ENV DB_PATH=/app/data/llm_shield.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
