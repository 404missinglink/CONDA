from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import faiss
import httpx
from app.embeddings import Provider, batch_embed_texts


@dataclass
class _Dummy:
    pass


class Searcher:
    def __init__(self, index_path: str, idmap_path: str):
        self.index = faiss.read_index(index_path)
        self.idmap = pd.read_parquet(idmap_path)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, List[str]]:
        q = query_vec.astype("float32")
        if q.ndim == 1:
            q = q[None, :]
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        rows = self.idmap.iloc[I[0]].to_dict(orient="records")
        return D[0], [r["text"] for r in rows]


async def embed_query(provider: Provider, text: str) -> np.ndarray:
    async with httpx.AsyncClient() as client:
        vecs = await provider.embed(client, [text])
    arr = np.asarray(vecs, dtype="float32")
    return arr




