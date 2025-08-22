# app_semrush_like.py
# 🚀 Semrush-like Keyword Explorer (Free) — Streamlit
# - Country selector for region-specific suggestions (Google/Bing/YouTube)
# - Enter one or many seeds
# - Multi-source autocomplete (Google, Bing, YouTube, Amazon*)
# - A–Z / 0–9 expansion, prefix/suffix expansion
# - Optional PAA + Related Searches (best-effort)
# - Long-tail filters, include/exclude lists, regex filters
# - Intent labeling + clustering
# - CSV export

import streamlit as st
import requests, time, re
from bs4 import BeautifulSoup
import pandas as pd
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

st.set_page_config(page_title="Semrush-like Keyword Explorer (Free)", page_icon="🧠", layout="wide")
st.title("🧠 Semrush-like Keyword Explorer (Free)")
st.caption("Tip: Start with a single seed to explore widely, then paste many seeds for batch mode.")

# ---------------------- Config ----------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
DELAY = 0.15  # polite delay between calls

# ---------------------- Suggest endpoints ----------------------
def google_suggest(q: str, country: str = "us"):
    """Google Autocomplete (unofficial). hl=language, gl=country."""
    try:
        r = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "firefox", "q": q, "hl": country.lower(), "gl": country.upper()},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            return data[1]
    except Exception:
        pass
    return []

def bing_suggest(q: str, country: str = "us"):
    """Bing Autocomplete (unofficial). mkt=en-COUNTRY."""
    try:
        r = requests.get(
            "https://api.bing.com/osjson.aspx",
            params={"query": q, "mkt": f"en-{country.upper()}"},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            return data[1]
    except Exception:
        pass
    return []

def youtube_suggest(q: str, country: str = "us"):
    """YouTube Autocomplete (unofficial). hl/gl like Google."""
    try:
        r = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "firefox", "ds": "yt", "q": q, "hl": country.lower(), "gl": country.upper()},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            return data[1]
    except Exception:
        pass
    return []

def amazon_suggest(q: str):
    """Amazon suggestions (region-dependent; may rate-limit)."""
    try:
        r = requests.get(
            "https://completion.amazon.com/api/2017/suggestions",
            params={"limit": 11, "prefix": q, "alias": "aps"},
            timeout=8,
            headers=HEADERS
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for s in data.get("suggestions", []):
            v = s.get("value")
            if v:
                out.append(v)
        return out
    except Exception:
        return []

def google_related_and_paa(q: str):
    """Best-effort scrape of Related Searches + People Also Ask (fragile selectors)."""
    out = []
    try:
        r = requests.get("https://www.google.com/search", params={"q": q}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Related searches — capture anchor text resembling related queries
        for a in soup.select("a"):
            txt = a.get_text(" ", strip=True)
            if txt and len(txt.split()) >= 2 and q.lower() not in txt.lower():
                href = a.get("href", "") or ""
                if "search?q=" in href:
                    out.append(txt)
        # PAA-ish — naive capture of question-like text
        for el in soup.find_all(text=True):
            t = str(el).strip()
            if t.endswith("?") and len(t.split()) >= 3:
                out.append(t)
    except Exception:
        pass
    return list(dict.fromkeys(out))

# ---------------------- Helpers ----------------------
ALPHANUM = list("abcdefghijklmnopqrstuvwxyz") + [str(i) for i in range(10)]
COMMON_SUFFIXES = ["for", "with", "without", "best", "cheap", "near me", "vs", "size", "guide", "ideas", "types", "comparison"]
COMMON_PREFIXES = ["best", "cheap", "top", "how to use", "types of", "difference between"]

def expansions(seed: str, breadth: int = 120):
    exps = [seed]
    # suffix A–Z/0–9
    exps += [f"{seed} {c}" for c in ALPHANUM]
    # prefix A–Z/0–9
    exps += [f"{c} {seed}" for c in ALPHANUM]
    # common suffixes/prefixes
    exps += [f"{seed} {s}" for s in COMMON_SUFFIXES]
    exps += [f"{p} {seed}" for p in COMMON_PREFIXES]
    # dedupe, cap
    return list(dict.fromkeys(exps))[:breadth]

def fetch_from_sources(query: str, sources: list[str], country: str = "us"):
    acc = []
    if "Google" in sources:
        acc += google_suggest(query, country=country)
        time.sleep(DELAY)
    if "Bing" in sources:
        acc += bing_suggest(query, country=country)
        time.sleep(DELAY)
    if "YouTube" in sources:
        acc += youtube_suggest(query, country=country)
        time.sleep(DELAY)
    if "Amazon" in sources:
        acc += amazon_suggest(query)
        time.sleep(DELAY)
    return acc

def classify_intent(kw: str) -> str:
    k = kw.lower()
    transactional = ["buy","price","prices","for sale","deal","deals","discount","near me","shop","order","sale","coupon","shipping"]
    commercial = ["best","top","review","compare","comparison","vs","alternative","alternatives","cheapest","brands"]
    informational = ["how to","what is","guide","tutorial","learn","ideas","types","size","benefits","pros","cons","difference"]
    if any(t in k for t in transactional): return "Transactional"
    if any(t in k for t in commercial): return "Commercial"
    if any(t in k for t in informational): return "Informational"
    if re.search(r"\b(for|with|without)\b", k): return "Commercial"
    return "Informational"

def cluster_keywords(kws: list[str], n_clusters: int, method: str = "auto"):
    if not kws:
        return [], 0
    use_embed = (method == "embeddings") or (method == "auto" and USE_EMBEDDINGS)
    if use_embed:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(kws)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = tfidf.fit_transform(kws)
    k = max(2, min(n_clusters, max(2, min(60, len(set(kws))//5))))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, k

def apply_filters(df: pd.DataFrame, min_words: int, max_words: int|None, include_terms, exclude_terms, regex_inc, regex_exc, long_tail_only: bool):
    def ok(kw: str):
        w = len(kw.split())
        if long_tail_only and w < 3: return False
        if w < min_words: return False
        if max_words and max_words > 0 and w > max_words: return False
        low = kw.lower()
        if include_terms:
            if not any(t in low for t in include_terms): return False
        if exclude_terms:
            if any(t in low for t in exclude_terms): return False
        if regex_inc:
            try:
                if not re.search(regex_inc, kw, flags=re.IGNORECASE): return False
            except Exception:
                pass
        if regex_exc:
            try:
                if re.search(regex_exc, kw, flags=re.IGNORECASE): return False
            except Exception:
                pass
        return True
    return df[df["Keyword"].apply(ok)].copy()

# ---------------------- UI ----------------------
with st.expander("⚙️ Settings & Sources", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sources = st.multiselect("Sources", ["Google","Bing","YouTube","Amazon"], default=["Google","Bing","YouTube"])
        breadth = st.slider("Expansion breadth per seed", 40, 240, 140, 10)
        min_per_seed = st.slider("Min suggestions per seed (best effort)", 10, 150, 40, 5)
        country = st.selectbox("Country (influences Google/Bing/YouTube)", ["us","au","in","uk","ca","de","fr","it","es","nl"], index=0)
    with c2:
        min_words = st.slider("Minimum words", 2, 6, 3, 1)
        max_words = st.slider("Maximum words (0 = no max)", 0, 10, 0, 1)
        long_tail_only = st.checkbox("Only long-tail (≥3 words)", True)
    with c3:
        include_terms = st.text_input("Must include terms (comma/line)", "")
        exclude_terms = st.text_area("Exclude terms (comma/line)", "bunnings\ntotal tools\nsydneytools\namazon\nebay")
        regex_inc = st.text_input("Regex include (optional)", "")
        regex_exc = st.text_input("Regex exclude (optional)", "")
    with c4:
        clustering_method = st.selectbox("Clustering method", ["auto (embeddings if available)","tfidf","embeddings"])
        cluster_target = st.slider("Target clusters", 2, 50, 12, 1)
        fetch_paa = st.checkbox("Enrich with Google Related & PAA (experimental)", False)

st.subheader("Seed Keywords")
tab1, tab2 = st.tabs(["Single seed (Semrush-style explore)","Multiple seeds (batch)"])
with tab1:
    seed_single = st.text_input("Enter a single seed keyword", placeholder="e.g. line trimmer")
with tab2:
    seeds_multi = st.text_area("Enter multiple seeds (one per line)", height=160, placeholder="e.g.\nline trimmer\ndrill\nimpact driver")

run_single = st.button("🔍 Explore Single Seed")
run_multi = st.button("🚀 Run Batch")

def expand_seed(seed: str):
    exps = expansions(seed, breadth=breadth)
    out_rows = []
    got = set()
    for ex in exps:
        suggs = fetch_from_sources(ex, sources, country=country)
        for s in suggs:
            s = s.strip()
            if s and s.lower() not in got:
                got.add(s.lower())
                out_rows.append({"Keyword": s, "Seed": seed, "SourceQuery": ex})
        if len(out_rows) >= min_per_seed:
            break
    # Optional enrichment
    if fetch_paa:
        more = google_related_and_paa(seed)
        for s in more:
            s = s.strip()
            if s and s.lower() not in got:
                got.add(s.lower())
                out_rows.append({"Keyword": s, "Seed": seed, "SourceQuery": seed + " (related/paa)"})
    return pd.DataFrame(out_rows)

def parse_list(s: str):
    if not s.strip():
        return []
    parts = re.split(r"[,\n]+", s.strip())
    return [p.strip().lower() for p in parts if p.strip()]

if run_single:
    if not seed_single.strip():
        st.warning("Please enter a seed keyword.")
        st.stop()
    with st.spinner("Fetching suggestions…"):
        df = expand_seed(seed_single.strip())
    if df.empty:
        st.warning("No suggestions found. Try different sources or increase breadth.")
        st.stop()

    # filters
    df = df.drop_duplicates(subset=["Keyword"]).reset_index(drop=True)
    inc = parse_list(include_terms)
    exc = parse_list(exclude_terms)
    df = apply_filters(df, min_words, max_words if max_words>0 else None, inc, exc, regex_inc, regex_exc, long_tail_only)

    if df.empty:
        st.warning("All suggestions were filtered out. Relax filters.")
        st.stop()

    # intent & clustering
    df["Intent"] = df["Keyword"].apply(classify_intent)
    labels, k = cluster_keywords(df["Keyword"].tolist(),
                                 n_clusters=cluster_target,
                                 method="auto" if clustering_method.startswith("auto") else clustering_method)
    df["Cluster"] = [f"Cluster {int(l)+1}" for l in labels]
    df = df.sort_values(["Cluster","Intent","Keyword"]).reset_index(drop=True)

    st.success(f"Collected {len(df)} suggestions across {k} clusters.")
    st.dataframe(df, use_container_width=True, height=520)

    # quick facets
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Intent counts**")
        st.write(df["Intent"].value_counts())
    with c2:
        st.write("**Top clusters**")
        st.write(df["Cluster"].value_counts().head(10))
    with c3:
        st.write("**Avg words per keyword**")
        st.write(round(df["Keyword"].str.split().map(len).mean(),2))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=f"keywords_{seed_single.replace(' ','_')}.csv", mime="text/csv")

if run_multi:
    seeds = [s.strip() for s in seeds_multi.splitlines() if s.strip()]
    if not seeds:
        st.warning("Please paste seeds in the batch box.")
        st.stop()
    all_rows = []
    with st.spinner("Fetching batch suggestions…"):
        for s in seeds:
            df_s = expand_seed(s)
            if not df_s.empty:
                all_rows.append(df_s)
    if not all_rows:
        st.warning("No suggestions found for the provided seeds.")
        st.stop()
    df = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["Keyword"])

    # filters
    inc = parse_list(include_terms)
    exc = parse_list(exclude_terms)
    df = apply_filters(df, min_words, max_words if max_words>0 else None, inc, exc, regex_inc, regex_exc, long_tail_only)

    if df.empty:
        st.warning("All suggestions were filtered out. Relax filters.")
        st.stop()

    # intent & clustering
    df["Intent"] = df["Keyword"].apply(classify_intent)
    labels, k = cluster_keywords(df["Keyword"].tolist(),
                                 n_clusters=cluster_target,
                                 method="auto" if clustering_method.startswith("auto") else clustering_method)
    df["Cluster"] = [f"Cluster {int(l)+1}" for l in labels]
    df = df.sort_values(["Seed","Cluster","Intent","Keyword"]).reset_index(drop=True)

    st.success(f"Collected {len(df)} total suggestions across {k} clusters.")
    st.dataframe(df, use_container_width=True, height=520)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="keywords_batch.csv", mime="text/csv")

st.markdown("---")
st.caption("⚠️ Endpoints are unofficial. Use modest rates and consider proxies for larger runs. Some sources (e.g., Amazon) are regionally restricted.")

st.caption(f"🌍 Country in use: **{country.upper()}**")

