"""Каждый клиент отправляет по одному трёхсекундному WAV каждые три секунды."""

import asyncio
from pathlib import Path
from statistics import quantiles
from time import monotonic

import httpx

AUDIO = Path(__file__).resolve().parent / "runs/loadtest-inputs-ru-ar/paced-3s"
SEGMENT_SECONDS = 3


def format_timings(latencies_s):
    if not latencies_s:
        return "н/д", "н/д"
    if len(latencies_s) == 1:
        p50 = p95 = p99 = latencies_s[0]
    else:
        cuts = quantiles(latencies_s, n=100, method="inclusive")
        p50, p95, p99 = cuts[49], cuts[94], cuts[98]
    rtfs = "/".join(f"{seconds / SEGMENT_SECONDS:.3f}" for seconds in (p50, p95, p99))
    return f"{p95 * 1000:.0f} мс", rtfs


async def stream(http, files, offset, start):
    errors = late = 0
    latencies_s = []
    for i in range(4):
        due = start + SEGMENT_SECONDS * i
        await asyncio.sleep(max(0, due - monotonic()))
        name = files[(offset + i) % len(files)].name
        sent = monotonic()
        try:
            response = await http.post("/predict", json={"path": f"/audio/{name}"})
            response.raise_for_status()
            latencies_s.append(monotonic() - sent)
        except httpx.HTTPError as exc:
            print(name, exc)
            errors += 1
        late += monotonic() > due + SEGMENT_SECONDS
    return errors, late, latencies_s


async def main():
    files = sorted(AUDIO.glob("*.wav"))
    if not files:
        raise SystemExit(f"Нет WAV в {AUDIO}")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as http:
        for clients in (1, 2, 4):
            start = monotonic()
            results = await asyncio.gather(*(stream(http, files, 4 * n, start) for n in range(clients)))
            latencies_s = [seconds for _, _, samples in results for seconds in samples]
            latency_p95, rtfs = format_timings(latencies_s)
            print(f"Клиентов: {clients}, запросов: {4 * clients}, "
                  f"ошибок: {sum(e for e, _, _ in results)}, "
                  f"позже {SEGMENT_SECONDS} с: {sum(late for _, late, _ in results)}, "
                  f"задержка p95: {latency_p95}, RTF p50/p95/p99: {rtfs}")


if __name__ == "__main__":
    asyncio.run(main())
