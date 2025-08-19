import asyncio
import os
import time
from dataclasses import dataclass
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import httpx
from dotenv import load_dotenv

from app.benchmark import Provider as BenchProvider, load_test, time_once
from app.search import Provider as SearchProvider, Searcher, embed_query


load_dotenv()
st.set_page_config(page_title="Embeddings Showdown: Speed Edition", layout="wide")


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


@st.cache_resource(show_spinner=False)
def load_searchers() -> dict[str, Searcher | None]:
    index_dir = get_env("INDEX_DIR", os.path.join("artifacts", "index"))
    idmap_path = os.path.join(index_dir, "id_map.parquet")
    searchers: dict[str, Searcher | None] = {"custom": None, "openai": None}
    paths = {
        "custom": os.path.join(index_dir, "conda_custom.faiss"),
        "openai": os.path.join(index_dir, "conda_openai.faiss"),
    }
    for key, path in paths.items():
        if os.path.exists(path) and os.path.exists(idmap_path):
            searchers[key] = Searcher(path, idmap_path)
    return searchers


def provider_configs():
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
    custom_url = os.environ.get(
        "CUSTOM_URL",
        "https://bsypq4hednykzclslp3aamimtm0ytpzd.lambda-url.eu-west-2.on.aws",
    )
    custom_model = os.environ.get("CUSTOM_MODEL", "takara-ai/m2v_science_v3c_clf")

    openai = None
    if openai_key and openai_key not in {"__unset__", "", "null", "None"}:
        openai = {
            "bench": BenchProvider("OpenAI", "openai", model=openai_model, openai_key=openai_key),
            "search": SearchProvider("OpenAI", "openai", model=openai_model, openai_key=openai_key),
        }
    custom = {
        "bench": BenchProvider("DS1", "custom", url=custom_url, model=custom_model),
        "search": SearchProvider("DS1", "custom", url=custom_url, model=custom_model),
    }
    return openai, custom


def percentile(xs: List[float], p: int) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1)))))
    return s[k]


def latency_race(query: str, searchers: dict[str, Searcher | None]):
    openai_p, custom_p = provider_configs()

    async def run_one(tag: str, p_search: SearchProvider, p_bench: BenchProvider):
        # Measure embed
        t0 = time.perf_counter()
        try:
            vec = await embed_query(p_search, query)
        except Exception as e:
            return {
                "provider": tag,
                "t_embed_ms": float("nan"),
                "t_search_ms": float("nan"),
                "t_end2end_ms": float("nan"),
                "first_hit": f"Error: {e}",
            }
        t1 = time.perf_counter()
        t_embed_ms = (t1 - t0) * 1000

        # Measure search using matching index
        s_key = "openai" if tag.lower().startswith("openai") else "custom"
        s_obj = searchers.get(s_key)
        if s_obj is None:
            return {
                "provider": tag,
                "t_embed_ms": t_embed_ms,
                "t_search_ms": float("nan"),
                "t_end2end_ms": float("nan"),
                "hits": [],
                "note": "No index available",
            }
        t2 = time.perf_counter()
        D, hits = s_obj.search(vec, top_k=5)
        t3 = time.perf_counter()
        t_search_ms = (t3 - t2) * 1000
        return {
            "provider": tag,
            "t_embed_ms": t_embed_ms,
            "t_search_ms": t_search_ms,
            "t_end2end_ms": (t3 - t0) * 1000,
            "hits": hits,
            "note": "",
        }

    tasks = []
    labels = []
    if openai_p is not None:
        tasks.append(run_one("OpenAI", openai_p["search"], openai_p["bench"]))
        labels.append("OpenAI")
    else:
        st.info("OPENAI_API_KEY not set; running only Custom provider.")
    tasks.append(run_one("DS1", custom_p["search"], custom_p["bench"]))
    labels.append("DS1")

    async def run_all():
        return await asyncio.gather(*tasks)
    results = asyncio.run(run_all())

    cols = st.columns(len(results))
    for col, res in zip(cols, results):
        with col:
            st.metric(f"{res['provider']} t_embed (ms)", f"{res['t_embed_ms']:.1f}")
            st.metric(f"{res['provider']} t_search (ms)", f"{res['t_search_ms']:.1f}")
            st.metric(f"{res['provider']} t_end2end (ms)", f"{res['t_end2end_ms']:.1f}")
            if res.get("note"):
                st.info(res["note"]) 
            if res.get("hits"):
                st.write("Top-5 retrieved:")
                for h in res["hits"][:5]:
                    st.caption(h)

    valid = [r for r in results if np.isfinite(r["t_end2end_ms"])]
    if len(valid) >= 2:
        winner = min(valid, key=lambda d: d["t_end2end_ms"])  # type: ignore
        st.success(f"Winner: {winner['provider']} (end-to-end)")


def build_index_from_idmap(provider: SearchProvider, index_dir: str, tag: str, batch_size: int = 32):
    idmap_path = os.path.join(index_dir, "id_map.parquet")
    if not os.path.exists(idmap_path):
        st.error(f"Missing id_map.parquet in {index_dir}. Build DS1 index first to create it.")
        return False
    df = pd.read_parquet(idmap_path)
    texts = df["text"].astype(str).tolist()

    async def embed_all():
        vecs: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                out = await provider.embed(client, batch)
                vecs.extend(out)
        return np.asarray(vecs, dtype="float32")

    embs = asyncio.run(embed_all())
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    out_path = os.path.join(index_dir, f"conda_{tag}.faiss")
    faiss.write_index(index, out_path)
    return True


def load_corpus_texts(index_dir: str, sample_size: int | None = None) -> list[str]:
    idmap_path = os.path.join(index_dir, "id_map.parquet")
    if os.path.exists(idmap_path):
        df = pd.read_parquet(idmap_path)
    else:
        csv_path = os.environ.get("CSV_PATH", os.path.join("data", "CONDA_train.csv"))
        df = pd.read_csv(csv_path, usecols=["utterance"]).dropna()
        # also write id_map for future runs
        out = pd.DataFrame({"row_id": range(len(df)), "text": df["utterance"].astype(str).tolist()})
        os.makedirs(index_dir, exist_ok=True)
        out.to_parquet(idmap_path, index=False)
    texts = df["text" if "text" in df.columns else "utterance"].astype(str).tolist()  # type: ignore
    if sample_size and sample_size > 0 and sample_size < len(texts):
        return texts[:sample_size]
    return texts


def reindex_and_time(provider: SearchProvider, tag: str, texts: list[str], index_dir: str, batch_size: int = 16):
    async def embed_all():
        vecs: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                out = await provider.embed(client, batch)
                vecs.extend(out)
        return np.asarray(vecs, dtype="float32")

    t0 = time.perf_counter()
    embs = asyncio.run(embed_all())
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    out_path = os.path.join(index_dir, f"conda_{tag}.faiss")
    faiss.write_index(index, out_path)
    t1 = time.perf_counter()
    return {"provider": tag.upper(), "seconds": t1 - t0, "dim": int(embs.shape[1]), "count": int(embs.shape[0])}


def load_section(texts: List[str]):
    st.subheader("Under Load")
    openai_p, custom_p = provider_configs()
    conc = st.slider("Concurrency", 1, 32, 8, step=1)
    dur = st.slider("Duration (s)", 5, 30, 20, step=5)
    if st.button("Run load test"):
        with st.spinner("Running load..."):
            tasks = []
            labels = []
            if openai_p is not None:
                tasks.append(load_test(openai_p["bench"], texts, conc, dur))
                labels.append("OpenAI")
            else:
                st.info("OPENAI_API_KEY not set; skipping OpenAI load test.")
            tasks.append(load_test(custom_p["bench"], texts, conc, dur))
            labels.append("DS1")
            async def run_all():
                return await asyncio.gather(*tasks)
            results = asyncio.run(run_all())
        rows = []
        for label, res in zip(labels, results):
            row = {"Provider": label}
            row.update(res)
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df)


def main():
    st.title("Embeddings Showdown: Speed Edition")
    searchers = load_searchers()

    st.sidebar.header("Settings")
    st.sidebar.write("Set OPENAI_API_KEY env var before running.")

    # Index management
    index_dir = get_env("INDEX_DIR", os.path.join("artifacts", "index"))
    openai_p, ds1_p = provider_configs()
    with st.expander("Indexes status / Build"):
        st.write(f"Index dir: {index_dir}")
        st.write(f"OpenAI index: {'present' if searchers.get('openai') else 'missing'}")
        st.write(f"DS1 index: {'present' if searchers.get('custom') else 'missing'}")
        
        # Show current corpus info
        current_texts = load_corpus_texts(index_dir)
        st.write(f"Current corpus size: {len(current_texts)} utterances")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Build OpenAI index", disabled=openai_p is None):
                ok = build_index_from_idmap(openai_p["search"], index_dir, tag="openai", batch_size=16)  # type: ignore
                if ok:
                    st.success("Built OpenAI index.")
                    st.rerun()
        with c2:
            if st.button("Build DS1 index"):
                ok = build_index_from_idmap(ds1_p["search"], index_dir, tag="custom", batch_size=16)  # type: ignore
                if ok:
                    st.success("Built DS1 index.")
                    st.rerun()
        
        # Full corpus rebuild option
        st.divider()
        st.write("**Rebuild with full dataset:**")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Rebuild OpenAI (full)", disabled=openai_p is None):
                with st.spinner("Building full OpenAI index..."):
                    texts = load_corpus_texts(index_dir, sample_size=None)
                    ok = build_index_from_idmap(openai_p["search"], index_dir, tag="openai", batch_size=16)  # type: ignore
                    if ok:
                        st.success(f"Built OpenAI index with {len(texts)} utterances.")
                        st.rerun()
        with c4:
            if st.button("Rebuild DS1 (full)"):
                with st.spinner("Building full DS1 index..."):
                    texts = load_corpus_texts(index_dir, sample_size=None)
                    ok = build_index_from_idmap(ds1_p["search"], index_dir, tag="custom", batch_size=16)  # type: ignore
                    if ok:
                        st.success(f"Built DS1 index with {len(texts)} utterances.")
                        st.rerun()

    with st.expander("Reindex speed test"):
        default_sample = 0  # 0 means full corpus
        sample = st.number_input("Sample size (0 = full corpus)", min_value=0, value=default_sample, step=500)
        batch = st.number_input("Batch size", min_value=4, max_value=128, value=16, step=4)
        if st.button("Run reindex timing"):
            texts = load_corpus_texts(index_dir, sample_size=sample if sample > 0 else None)
            tasks = []
            labels = []
            if openai_p is not None:
                tasks.append(reindex_and_time(openai_p["search"], tag="openai", texts=texts, index_dir=index_dir, batch_size=batch))  # type: ignore
                labels.append("OpenAI")
            tasks.append(reindex_and_time(ds1_p["search"], tag="custom", texts=texts, index_dir=index_dir, batch_size=batch))  # type: ignore
            # run sequentially to avoid rate-limit thrash
            results = []
            for t in tasks:
                results.append(t)
            df = pd.DataFrame(results)
            st.dataframe(df)
            if len(results) >= 2:
                fastest = min(results, key=lambda r: r["seconds"])  # type: ignore
                st.success(f"Fastest reindex: {fastest['provider']} ({fastest['seconds']:.2f}s)")

    example = "report player trolling after lane grief"
    query = st.text_input("Query", value=example)
    if st.button("Latency Race"):
        latency_race(query, searchers)

    st.divider()
    st.subheader("Tail Latency Snapshot & Throughput Gauge")
    texts = [
        "he keeps feeding mid",
        "report this guy",
        "gg ez",
        "good luck have fun",
        "stop griefing",
        "push top now",
    ]
    load_section(texts)


if __name__ == "__main__":
    main()


