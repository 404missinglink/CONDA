import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

import httpx
from app.embeddings import Provider


async def time_once(p: Provider, text: str) -> Dict:
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        vecs = await p.embed(client, [text])
        t1 = time.perf_counter()
        return {"ok": True, "t_embed_ms": (t1 - t0) * 1000.0, "dim": len(vecs[0])}


async def load_test(
    p: Provider, texts: List[str], concurrency: int = 8, duration_s: int = 20
) -> Dict[str, float | int]:
    lat: List[float] = []
    errs = 0

    async def worker():
        nonlocal errs
        i = 0
        async with httpx.AsyncClient() as client:
            t_end = time.perf_counter() + duration_s
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                try:
                    await p.embed(client, [texts[i % len(texts)]])
                    lat.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    errs += 1
                i += 1

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    lat.sort()

    def perc(pct: int) -> float:
        if not lat:
            return float("inf")
        k = int(max(0, min(len(lat) - 1, round(pct / 100 * (len(lat) - 1)))))
        return lat[k]

    return {
        "count": len(lat),
        "errors": errs,
        "p50": perc(50),
        "p90": perc(90),
        "p95": perc(95),
        "p99": perc(99),
        "qps": len(lat) / duration_s if duration_s > 0 else 0.0,
    }




