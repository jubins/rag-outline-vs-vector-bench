FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bench ./bench
COPY llm_stub ./llm_stub

RUN useradd --create-home bench
USER bench

EXPOSE 8000
CMD ["uvicorn", "bench.main:app", "--host", "0.0.0.0", "--port", "8000"]
