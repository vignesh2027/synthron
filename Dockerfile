FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

RUN mkdir -p /app/data/chroma /app/data/episodic /app/logs

ENV PYTHONUNBUFFERED=1
ENV SYNTHRON_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "synthron.api.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
