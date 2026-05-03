# syntax=docker/dockerfile:1.7

# --- builder stage: install only production dependencies ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements-prod.txt .
RUN pip install --user --no-cache-dir -r requirements-prod.txt

# --- runtime stage: minimal image with only what /predict needs ---
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/app/.local/bin:$PATH

# Run as non-root for security
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app

COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Application code
COPY --chown=app:app domain/ domain/
COPY --chown=app:app app/ app/
COPY --chown=app:app infrastructure/ infrastructure/
COPY --chown=app:app interfaces/ interfaces/
COPY --chown=app:app main.py .

# Trained ML artifacts (must exist locally before building — they are .gitignored)
COPY --chown=app:app misc/model.onnx misc/scaler.joblib misc/

EXPOSE 8000

# Database credentials must be supplied at runtime via -e or a secret manager.
# Do NOT bake .env into the image.
CMD ["uvicorn", "interfaces.http.api:app", "--host", "0.0.0.0", "--port", "8000"]
