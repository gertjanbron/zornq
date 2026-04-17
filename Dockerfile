# ZornQ reproducible container image
# Pinned on Python 3.12-slim (Debian bookworm); mirror of tested-working stack.
#
# Build:
#   docker build -t zornq:latest .
#
# Run unit tests:
#   docker run --rm zornq:latest pytest code -q
#
# Run interactive:
#   docker run --rm -it -v "$(pwd)/results:/app/results" zornq:latest bash
#
# Run a benchmark (output blijft persistent via volume-mount):
#   docker run --rm -v "$(pwd)/results:/app/results" zornq:latest \
#       python -u code/b176_benchmark.py
#
# Voor GPU-pad: gebruik `nvidia/cuda:12.2.0-runtime-ubuntu22.04` als base-image
# en voeg `pip install cupy-cuda12x` toe. Losse Dockerfile (`Dockerfile.gpu`) komt
# mee onder B11b follow-up indien daadwerkelijk vereist.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Systeem-dependencies voor wetenschappelijke Python (scipy/cvxpy compileren soms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Reproduceerbare Python-timezone + locale-instellingen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=0 \
    TZ=UTC \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /app

# Stap 1: eerst alleen de dependencies-bestanden kopiëren (cache-vriendelijk)
COPY requirements.txt pyproject.toml ./

# Stap 2: installeer dependencies - deze layer wordt alleen opnieuw gebouwd als
# requirements.txt of pyproject.toml verandert.
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install pytest pytest-cov

# Stap 3: kopieer de daadwerkelijke code
# (broncode wordt apart gekopieerd zodat dependency-layer herbruikbaar blijft)
COPY code/ ./code/
COPY docs/ ./docs/

# Default-commando: draai de complete test-suite.
CMD ["pytest", "code", "-q", "--tb=short"]
