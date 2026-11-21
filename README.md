# AudioLID

CPU-сервис определяет язык аудио: `ru`, `ar` или `unknown`. Обученная модель уже в `models/`. Команды выполняйте из корня проекта.

## Сервис и запрос

Соберите образ и запустите сервис в фоне.

```sh
docker build -t audiolid:local .
docker run -d --rm --name audiolid-local --cpus=4 -p 127.0.0.1:8000:8000 \
  -v "$PWD/runs/loadtest-inputs-ru-ar/paced-3s:/audio:ro" audiolid:local
```

Дождитесь ответа `{"status":"ok"}` и отправьте путь к WAV внутри контейнера.

```sh
curl -fsS --retry 15 --retry-all-errors --retry-delay 1 http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"path":"/audio/002_common_voice_ar_24422744.wav"}'
```

После тестов остановите сервис командой `docker stop audiolid-local`.

## Тесты

```sh
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install httpx && python loadtest.py                 # 1, 2 и 4 клиента при работающем сервисе
```

## Обучение в Kaggle (по желанию)

1. Подключите датасеты `manofwar/commonvoice-ar-ru`, `manofwar/mozila-id-et-el-fi` и `dogrose/musan-dataset`. Включите Internet для загрузки ECAPA, установите `%pip install speechbrain==1.1.2` и выполните [data_preparation.ipynb](data_preparation.ipynb).
2. Сохранив содержимое `/kaggle/working`, выполните [train2.ipynb](train2.ipynb). Для обновления сервиса скопируйте `xgb_model.json` и `xgb_summary.json` из `/kaggle/working/ecapa_results/` в `models/` и пересоберите образ.

[Отчёт о качестве и нагрузке](AudioLID.md)
