"""HTTP-сервис AudioLID: POST /predict {"path": "..."} → {"label": ..., "probabilities": ...}.

Запуск: uvicorn service.app:build_app --factory --host 0.0.0.0 --port 8000
"""

import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service.inference import AudioError


class PredictRequest(BaseModel):
    path: str


def create_app(model, workers=4):
    """model — объект с predict(path) → dict; одновременно считается не больше `workers` запросов."""
    app = FastAPI(title="AudioLID")
    slots = threading.BoundedSemaphore(workers)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest):  # обычная функция: FastAPI выполняет её в пуле потоков
        try:
            with slots:
                return model.predict(request.path)
        except FileNotFoundError:
            raise HTTPException(404, f"файл не найден: {request.path}") from None
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from None

    return app


def build_app():
    """Точка входа uvicorn: модель из MODEL_DIR, WORKERS одновременных запросов."""
    from service.inference import LanguageID

    model = LanguageID(os.environ.get("MODEL_DIR", "models"))
    return create_app(model, workers=int(os.environ.get("WORKERS", "4")))
