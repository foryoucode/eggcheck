FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# CPU-only PyTorch: avoid multi-GB Nvidia CUDA wheels on a regular VPS.
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' \
 && pip install -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
COPY simulator ./simulator
COPY models ./models
RUN useradd -m -u 10001 eggcheck && chown -R eggcheck:eggcheck /app
USER eggcheck
EXPOSE 8000
HEALTHCHECK --interval=40s --timeout=8s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=5)" || exit 1
CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
