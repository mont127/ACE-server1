FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (tokenizers/transformers often need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install torch (CPU) first to avoid resolver weirdness
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY ace_server.py /app/ace_server.py
COPY static /app/static
RUN mkdir -p /app/mem

EXPOSE 8000

CMD ["uvicorn", "ace_server:app", "--host", "0.0.0.0", "--port", "8000"]