# app.py
# Streamlit Long-Tail Keyword Finder

import streamlit as st
import requests, time, re, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Optional embeddings
USE_EMBEDDINGS = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    USE_EMBEDDINGS = True
except Exception:
    pass

SUGGEST_URL = "https://suggestqueries.google.com/complete/search?client=firefox&q={q}"

def google_suggest(q: str):
    try:
        r = requests.get(SUGGEST_URL.format(q=q), timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            return data[1]
    except Exception:
        return []
    return []

def expand_queries(seed: str, modifiers: list[str], hard_min: int = 20, max_expansions: int = 120):
    seen = set()
    results = []
    expansions = [seed.strip()]
    for m in modifiers:
        expansions.append(f"{seed} {m}")
    expansions = list(dict.fromkeys(expansions))[:max_expansions]

    for exp in expansions:
        suggs = google_suggest(exp)
        for s in suggs:
            if len(s.split()) >= 3:
                results.append(s)
        time.sleep(0.2)
        if len(results) >= hard_min:
            break
    return results

def classify_intent(kw: str) -> str:
    k = kw.lower()
    transactional = ["buy","price","sale","order","shop","discount","near me"]
    commercial = ["best","top","review","compare","vs","cheap"]
    informational = ["how to","guide","what is","ideas","types","benefits"]
    if any(t in k for t in transactional): return "Transactional"
    if any(t in k for t in commercial): return "Commercial"
    if any(t in k for t in informational): return "Informational"
    return "Informational"

def cluster_keywords(keywords: list[str], n_clusters: int = 10, method: str = "auto"):
    if not keywords:
        return [], 0
    use_embed = (method == "embeddings") or (method == "auto" and USE_EMBEDDINGS)
    if use_embed:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(keywords)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = tfidf.fit_transform(keywords)
    k = max(2, min(n_clusters, max(2, min(50, len(set(keywords))//5))))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, k

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Long-Tail Keyword Finder", page_icon="🔎", layout="wide")
st.title("🔎 Long-Tail Keyword Finder")

seed_text = st.text_area("Enter seed keywords (one per line)", height=200)
run = st.button("🚀 Start Analysis")

if run:
    seeds = [s.strip() for s in seed_text.splitlines() if s.strip()]
    if not seeds:
        st.warning("Please provide at least one seed keyword.")
        st.stop()

    modifiers = list("abcdefghijklmnopqrstuvwxyz") + [str(i) for i in range(10)] + ["for","with","best","cheap","guide"]

    rows = []
    for seed in seeds:
        kws = expand_queries(seed, modifiers)
        for kw in kws:
            rows.append({"Keyword": kw, "Seed": seed})

    df = pd.DataFrame(rows).drop_duplicates(subset=["Keyword"]).reset_index(drop=True)
    df["Intent"] = df["Keyword"].apply(classify_intent)
    labels, k = cluster_keywords(df["Keyword"].tolist(), n_clusters=12)
    df["Cluster"] = [f"Cluster {int(l)+1}" for l in labels]

    st.success(f"Found {len(df)} keywords across {k} clusters.")
    st.dataframe(df, use_container_width=True, height=500)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="long_tail_keywords.csv", mime="text/csv")
