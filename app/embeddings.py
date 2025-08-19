from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import List

import httpx
import numpy as np


@dataclass
class Provider:
    name: str
    kind: str  # "openai" or "custom"
    url: str | None = None
    model: str | None = None
    openai_key: str | None = None

    async def embed(self, client: httpx.AsyncClient, texts: List[str]) -> List[List[float]]:
        max_retries = int(os.environ.get("EMBED_MAX_RETRIES", "4"))
        base_delay = float(os.environ.get("EMBED_BASE_DELAY", "0.3"))
        for attempt in range(max_retries):
            try:
                if self.kind == "openai":
                    r = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.openai_key}"},
                        json={"model": self.model, "input": texts},
                        timeout=45.0,
                    )
                    r.raise_for_status()
                    data = r.json()
                    return [d.get("embedding") for d in data["data"]]
                else:
                    base = (self.url or "").rstrip("/")
                    endpoint = base + "/v1/embeddings" if not base.endswith("/v1/embeddings") else base
                    r = await client.post(endpoint, json={"model": self.model, "input": texts}, timeout=45.0)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and "data" in data:
                        return [d.get("embedding") for d in data["data"]]
                    if "embeddings" in data:
                        return data["embeddings"]
                    raise ValueError("Unexpected embeddings response shape")
            except Exception:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.random() * 0.1
                await asyncio.sleep(delay)


async def batch_embed_texts(provider: Provider, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vectors: List[List[float]] = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = await provider.embed(client, batch)
            vectors.extend(emb)
    return np.asarray(vectors, dtype="float32")


