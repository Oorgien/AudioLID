"""Модель AudioLID: аудиофайл → ru / ar / unknown.

Повторяет рабочие ноутбуки: load_clip из data_preparation.ipynb (центральные 3 с, моно,
8 кГц → 16 кГц), 512 признаков замороженной ECAPA (classifier act → norm → DNN),
XGBoost и порог из train2.ipynb.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import xgboost as xgb
from scipy.signal import resample_poly

SECONDS = 3          # фрагмент из центра записи, как при обучении
MIN_DURATION = 0.05  # ECAPA нужно несколько кадров звука; запись короче — unknown


class AudioError(ValueError):
    """Файл есть, но не читается как аудио."""


def load_audio(path, seconds=SECONDS):
    """Центральные `seconds` секунд записи, моно, 8 кГц → 16 кГц.

    Возвращает (float32-сигнал 16 кГц, длительность взятого фрагмента в секундах).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with sf.SoundFile(path) as audio:
            rate = audio.samplerate
            frames = int(seconds * rate)
            if audio.format == "MP3":  # длина MP3 в заголовке приблизительная: декодируем целиком
                samples = audio.read(dtype="float32", always_2d=True)
                offset = max(0, (len(samples) - frames) // 2)
                samples = samples[offset:offset + frames]
            else:
                audio.seek(max(0, (len(audio) - frames) // 2))
                samples = audio.read(frames, dtype="float32", always_2d=True)
    except sf.SoundFileError as exc:
        raise AudioError(f"не удалось прочитать аудио: {exc}") from None
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    phone = resample_poly(samples.mean(axis=1), 8000, rate)
    return resample_poly(phone, 2, 1).astype(np.float32), len(samples) / rate


def decide(probabilities, labels, threshold):
    """Argmax; если победил ru или ar, но его вероятность ниже порога — unknown."""
    best = int(np.argmax(probabilities))
    if labels[best] != "unknown" and probabilities[best] < threshold:
        return "unknown"
    return labels[best]


class LanguageID:
    """ECAPA из model_dir/ecapa, XGBoost из model_dir/xgb_model.json, порог и метки из xgb_summary.json."""

    def __init__(self, model_dir):
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        model_dir = Path(model_dir).resolve()
        summary = json.loads((model_dir / "xgb_summary.json").read_text(encoding="utf-8"))
        self.labels = list(summary["labels"])
        self.threshold = float(summary["threshold"])

        torch.set_num_threads(1)  # по ядру на запрос: параллельность даёт число одновременных запросов
        self.booster = xgb.Booster(model_file=str(model_dir / "xgb_model.json"))
        self.booster.set_param({"nthread": 1})
        ecapa = model_dir / "ecapa"
        self.ecapa = EncoderClassifier.from_hparams(
            source=str(ecapa), savedir=tempfile.mkdtemp(prefix="audiolid-ecapa-"),
            overrides={"pretrained_path": str(ecapa)}, run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.NO_LINK, fetch_config=FetchConfig(allow_network=False),
        ).eval()

        # Пробный прогон: несовместимые файлы модели ломают запуск, а не первый запрос.
        probabilities = self.booster.inplace_predict(self.features(np.zeros(16000 * SECONDS, np.float32))[None])
        if probabilities.shape != (1, len(self.labels)):
            raise ValueError(f"XGBoost выдаёт {probabilities.shape[1:]} вероятностей, а меток {len(self.labels)}")

    @torch.inference_mode()
    def features(self, waveform):
        """512 признаков на входе головы ECAPA: эмбеддинг → act → norm → DNN."""
        classifier = self.ecapa.mods.classifier
        embedding = self.ecapa.encode_batch(torch.from_numpy(waveform)[None])  # [1, 1, 256]
        return classifier.DNN(classifier.norm(classifier.act(embedding)))[0, 0].numpy()

    def predict(self, path):
        """{"label": ru|ar|unknown, "probabilities": {метка: p} или None для слишком коротких записей}."""
        waveform, duration = load_audio(path)
        if duration < MIN_DURATION:
            return {"label": "unknown", "probabilities": None}
        probabilities = self.booster.inplace_predict(self.features(waveform)[None])[0]
        return {
            "label": decide(probabilities, self.labels, self.threshold),
            "probabilities": {label: float(p) for label, p in zip(self.labels, probabilities)},
        }
