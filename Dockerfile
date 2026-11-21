FROM python:3.12-slim

# По одному потоку на библиотеку: параллельность дают одновременные запросы (WORKERS),
# каждый занимает одно ядро. Запуск с ограничением: docker run --cpus=4 ...
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MODEL_DIR=/app/models \
    WORKERS=4

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0 \
 && pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && python -c "import torchaudio"

COPY service/ service/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "service.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
