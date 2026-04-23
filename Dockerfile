# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 on Debian slim — small footprint, compatible with all dependencies
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# libgomp1      → OpenMP, required by ONNX Runtime (used by Docling OCR models)
# libgl1        → OpenCV image processing (used by Docling layout analysis)
# libglib2.0-0  → GLib, required by OpenCV
# supervisor    → process manager to run FastAPI + Streamlit together
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first (before code) so Docker can cache this layer.
# If only your code changes, pip install is skipped on rebuild — saves ~5 min.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=200 --retries=5 -r requirements.txt

# ── Pre-download Docling models ────────────────────────────────────────────────
# Docling downloads layout + OCR models (~500MB) from HuggingFace on first use.
# We do this at BUILD time so the container starts instantly.
# If done at runtime, the first upload would take 5-10 minutes — unacceptable.
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# ── Copy application code ──────────────────────────────────────────────────────
# Done AFTER pip install and model download so code changes don't invalidate
# the expensive dependency and model layers above.
COPY . .

# ── Supervisor config ──────────────────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Port ──────────────────────────────────────────────────────────────────────
# HF Spaces exposes port 7860 — Streamlit listens here.
# FastAPI runs on 8000 internally (not exposed to internet, only within container).
EXPOSE 7860

# ── Start ─────────────────────────────────────────────────────────────────────
# supervisord starts both FastAPI and Streamlit as defined in supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
