FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /workspace/backend

COPY backend/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY backend/app ./app

RUN chown -R appuser:appuser /workspace

USER appuser

EXPOSE 8000

CMD [\
    "uvicorn",\
    "app.main:app",\
    "--host", "0.0.0.0",\
    "--port", "8000",\
    "--workers", "2"\
]