import os
import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import faiss
import httpx
from app.embeddings import Provider, batch_embed_texts


DEFAULT_SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "0"))  # 0 = full corpus
DEFAULT_TOPK = 10


@dataclass
class _Dummy:
    pass


def load_corpus(csv_path: str, sample_size: int | None) -> List[Tuple[int, str]]:
    df = pd.read_csv(csv_path)
    if "utterance" not in df.columns:
        raise ValueError("Expected column 'utterance' in the dataset")
    df = df.dropna(subset=["utterance"])  # type: ignore
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return [(int(i), str(t)) for i, t in zip(df.index, df["utterance"].tolist())]


# uses shared batch_embed_texts from app.embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, id_map: List[Tuple[int, str]], out_dir: str, tag: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, f"conda_{tag}.faiss")
    ids_path = os.path.join(out_dir, "id_map.parquet")
    faiss.write_index(index, index_path)
    pd.DataFrame(id_map, columns=["row_id", "text"]).to_parquet(ids_path, index=False)


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    import asyncio

    csv_path = os.environ.get("CSV_PATH", os.path.join("data", "CONDA_train.csv"))
    sample_size = int(os.environ.get("SAMPLE_SIZE", str(DEFAULT_SAMPLE_SIZE)))
    out_dir = os.environ.get("INDEX_DIR", os.path.join("artifacts", "index"))

    custom_url = os.environ.get(
        "CUSTOM_URL",
        "https://bsypq4hednykzclslp3aamimtm0ytpzd.lambda-url.eu-west-2.on.aws",
    )
    custom_model = os.environ.get("CUSTOM_MODEL", "takara-ai/m2v_science_v3c_clf")
    openai_model = os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
    openai_key = os.environ.get("OPENAI_API_KEY")

    use_provider = os.environ.get("INDEX_PROVIDER", "custom")

    if use_provider not in ("custom", "openai"):
        raise SystemExit("INDEX_PROVIDER must be 'custom' or 'openai'")

    provider = Provider(
        name=use_provider,
        kind=use_provider,
        url=custom_url if use_provider == "custom" else None,
        model=custom_model if use_provider == "custom" else openai_model,
        openai_key=openai_key,
    )

    _print(f"Loading corpus from {csv_path} (sample_size={sample_size if sample_size > 0 else 'full'}) …")
    corpus = load_corpus(csv_path, sample_size if sample_size > 0 else None)
    texts = [t for _, t in corpus]

    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    _print(
        f"Embedding {len(texts)} texts with provider={use_provider} (batch_size={batch_size}) …"
    )
    t0 = time.perf_counter()
    try:
        embeddings = asyncio.run(batch_embed_texts(provider, texts, batch_size=batch_size))
    except Exception as e:
        if use_provider == "custom" and openai_key:
            _print(
                f"Custom provider failed ({e}); falling back to OpenAI model={openai_model} …"
            )
            provider_fallback = Provider(
                name="openai",
                kind="openai",
                model=openai_model,
                openai_key=openai_key,
            )
            embeddings = asyncio.run(
                batch_embed_texts(provider_fallback, texts, batch_size=max(8, batch_size // 2))
            )
        else:
            raise
    t1 = time.perf_counter()
    _print(f"Embeddings shape: {embeddings.shape}; time={(t1 - t0):.2f}s")

    _print("Building FAISS index …")
    index = build_faiss_index(embeddings)

    _print(f"Saving index to {out_dir} …")
    save_index(index, corpus, out_dir, tag=provider.kind)
    _print("Done.")


if __name__ == "__main__":
    main()




