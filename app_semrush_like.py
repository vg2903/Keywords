# app_semrush_like.py
# 🚀 Semrush-like Keyword Explorer (Free) — Streamlit
# - Country selector (Google/Bing/YouTube)
# - Attribute-driven refinement (brands/specs/features/units)
# - Preset packs + Data sources (Manual / CSV upload / Google Sheet CSV)
# - Templates to manufacture structured long-tails
# - Numeric range DSL to auto-generate specs
# - Synonyms (canonicalization/expansion)
# - Multi-source validation (consensus)
# - Filters, intent, clustering, CSV export

import streamlit as st
import requests, time, re
from functools import lru_cache
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
st.caption("Start with a single seed to explore widely, then paste many seeds for batch mode.")

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
    exps += [f"{seed} {c}" for c in ALPHANUM]   # suffix A–Z/0–9
    exps += [f"{c} {seed}" for c in ALPHANUM]   # prefix A–Z/0–9
    exps += [f"{seed} {s}" for s in COMMON_SUFFIXES]
    exps += [f"{p} {seed}" for p in COMMON_PREFIXES]
    return list(dict.fromkeys(exps))[:breadth]

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
        if include_terms and not any(t in low for t in include_terms): return False
        if exclude_terms and any(t in low for t in exclude_terms): return False
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

def _to_list(txt: str):
    return [t.strip().lower() for t in re.split(r"[,\n]+", txt) if t.strip()]

def generate_attribute_candidates(seed: str, brands: list[str], specs: list[str], features: list[str]) -> list[str]:
    seed = seed.strip()
    cands = set()
    for b in brands:
        for s in specs:
            cands.add(f"{b} {s} {seed}")
            cands.add(f"{s} {b} {seed}")
    for b in brands:
        cands.add(f"{b} {seed}")
        for f in features:
            cands.add(f"{b} {seed} {f}")
            cands.add(f"{b} {f} {seed}")
    for s in specs:
        cands.add(f"{s} {seed}")
    for f in features:
        cands.add(f"{f} {seed}")
        cands.add(f"{seed} {f}")
    return list(cands)

def count_attribute_hits(text: str, tokens: list[str]) -> int:
    tl = text.lower()
    return sum(1 for t in tokens if t in tl)

def has_number(text: str) -> bool:
    return bool(re.search(r"\b\d+\b", text))

def has_unit(text: str, units: list[str]) -> bool:
    tl = text.lower()
    return any(re.search(rf"\b{re.escape(u)}\b", tl) for u in units)

# ---------------------- Caching & validation ----------------------
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

@lru_cache(maxsize=20000)
def cached_suggest_single_source(query: str, source: str, country: str):
    if source == "Google":
        return tuple(google_suggest(query, country=country))
    if source == "Bing":
        return tuple(bing_suggest(query, country=country))
    if source == "YouTube":
        return tuple(youtube_suggest(query, country=country))
    if source == "Amazon":
        return tuple(amazon_suggest(query))
    return tuple()

def validate_keyword(phrase: str, country: str, require_sources=2, sources=("Google","Bing","YouTube")) -> bool:
    """Keep only if the exact phrase is confirmed by >= require_sources among the given sources."""
    hits = 0
    plow = phrase.strip().lower()
    for src in sources:
        suggs = cached_suggest_single_source(phrase, src, country)
        if any(plow == s.strip().lower() for s in suggs):
            hits += 1
        if hits >= require_sources:
            return True
    return False

# ---------------------- UI: Settings & Sources ----------------------
with st.expander("⚙️ Settings & Sources", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sources = st.multiselect("Sources", ["Google","Bing","YouTube","Amazon"], default=["Google","Bing","YouTube"])
        breadth = st.slider("Expansion breadth per seed", 40, 240, 140, 10)
        min_per_seed = st.slider("Min suggestions per seed (best effort)", 10, 150, 40, 5)
        country = st.selectbox("Country (affects Google/Bing/YouTube)", ["us","au","in","uk","ca","de","fr","it","es","nl"], index=0)
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

# ---------------------- UI: Data sources (attributes at scale) ----------------------
with st.expander("📚 Data sources (attributes at scale)", expanded=False):
    mode = st.radio("Load attributes from", ["Manual lists", "Upload CSV", "Google Sheet (CSV URL)"], horizontal=True)
    uploaded_attrs = None
    sheet_url = ""
    if mode == "Upload CSV":
        uploaded_attrs = st.file_uploader("Upload CSV with columns: type,value   (type in: brand, spec, feature, unit)", type=["csv"])
    elif mode == "Google Sheet (CSV URL)":
        sheet_url = st.text_input("Paste CSV export URL of your Google Sheet")

def load_attributes_from_csv(file_or_url):
    df = pd.read_csv(file_or_url)
    # Expect columns: type,value. type in {"brand","spec","feature","unit"}
    out = {"brand": set(), "spec": set(), "feature": set(), "unit": set()}
    if "type" in df.columns and "value" in df.columns:
        for _, row in df.iterrows():
            t = str(row["type"]).strip().lower()
            v = str(row["value"]).strip().lower()
            if t in out and v:
                out[t].add(v)
    return {k: sorted(v) for k, v in out.items()}

# ---------------------- UI: Attribute-driven refinement panel ----------------------
with st.expander("🎯 Attribute-driven refinement (brands/specs/features)", expanded=True):
    colA, colB, colC, colD = st.columns(4)

    # Presets
    PRESETS = {
        "None": {
            "brands": "",
            "specs": "",
            "features": "",
            "units": "v, volt, volts, gauge, mm, inch"
        },
        "Power Tools": {
            "brands": "makita\nhusqvarna\nmilwaukee\ndewalt\ngreenworks\nbosch\nryobi\nhitachi\nhilti",
            "specs": "12v\n18v\n24v\n36v\n40v\n54v\n16 gauge\n18 gauge\n23 gauge",
            "features": "cordless\ncorded\nair\npneumatic\nbrushless\ncompact\nframing\nfinish\nbrad",
            "units": "v, volt, volts, gauge, mm, inch, ah"
        },
        "Garden Tools": {
            "brands": "husqvarna\nmakita\ndewalt\nstihl\nego\ngreenworks\nryobi",
            "specs": "18v\n36v\n40v\n58v\n2 stroke\n4 stroke\n25cc\n30cc\n40cc",
            "features": "cordless\nbattery operated\nmetal blade\nline\nstring\nwith wheels\nstraight shaft",
            "units": "v, volt, volts, cc, stroke, inch, mm"
        }
    }

    with colA:
        use_attr_mode = st.checkbox("Use attribute combinator", True)
        min_attrs_required = st.slider("Require at least N attributes", 1, 3, 1, 1)
        preset_choice = st.selectbox("Preset pack", list(PRESETS.keys()), index=1)
        apply_preset = st.button("Apply preset")

    # Initialize session_state for text areas
    if "brands_text" not in st.session_state:
        st.session_state["brands_text"] = PRESETS["Power Tools"]["brands"]
    if "specs_text" not in st.session_state:
        st.session_state["specs_text"] = PRESETS["Power Tools"]["specs"]
    if "features_text" not in st.session_state:
        st.session_state["features_text"] = PRESETS["Power Tools"]["features"]
    if "units_text" not in st.session_state:
        st.session_state["units_text"] = PRESETS["Power Tools"]["units"]

    if apply_preset:
        st.session_state["brands_text"] = PRESETS[preset_choice]["brands"]
        st.session_state["specs_text"] = PRESETS[preset_choice]["specs"]
        st.session_state["features_text"] = PRESETS[preset_choice]["features"]
        st.session_state["units_text"] = PRESETS[preset_choice]["units"]

    with colB:
        require_number = st.checkbox("Require a number (e.g., 16, 36, 40)", True)
        require_unit = st.checkbox("Require a unit/spec (e.g., v, gauge, mm)", True)

    with colC:
        brands_text = st.text_area("Brands (one per line)", st.session_state["brands_text"], height=160, key="brands_text")
        specs_text = st.text_area("Specs (one per line)", st.session_state["specs_text"], height=160, key="specs_text")

    with colD:
        features_text = st.text_area("Features (one per line)", st.session_state["features_text"], height=160, key="features_text")
        units_text = st.text_input("Units (comma-separated)", st.session_state["units_text"], key="units_text")

# ---------------------- UI: Templates (power users) ----------------------
with st.expander("🧩 Templates (power user)", expanded=False):
    templates_text = st.text_area(
        "One template per line. Use {seed}, {brand}, {spec}, {feature}.\n"
        "Examples:\n"
        "{brand} {spec} {seed}\n{spec} {seed} for {feature}\n{brand} {seed}\n{spec} {seed}",
        "{brand} {spec} {seed}\n{spec} {seed}\n{brand} {seed} {feature}",
        height=140
    )

def render_templates(templates, seed, brands, specs, features, limit_per_template=5000):
    out = set()
    for tpl in templates:
        tpl = tpl.strip()
        if not tpl:
            continue
        has_brand   = "{brand}"   in tpl
        has_spec    = "{spec}"    in tpl
        has_feature = "{feature}" in tpl

        b_iter = brands   if has_brand   else [None]
        s_iter = specs    if has_spec    else [None]
        f_iter = features if has_feature else [None]

        count = 0
        for b in b_iter:
            for s in s_iter:
                for f in f_iter:
                    phrase = tpl.replace("{seed}", seed)
                    if b is not None: phrase = phrase.replace("{brand}", b)
                    if s is not None: phrase = phrase.replace("{spec}", s)
                    if f is not None: phrase = phrase.replace("{feature}", f)
                    phrase = re.sub(r"\s+", " ", phrase).strip()
                    out.add(phrase)
                    count += 1
                    if count >= limit_per_template:
                        break
                if count >= limit_per_template: break
            if count >= limit_per_template: break
    return list(out)

# ---------------------- UI: Auto-generate specs (ranges & lists) ----------------------
with st.expander("🔢 Auto-generate specs (ranges & lists)", expanded=False):
    gen_specs = st.text_area(
        "Describe ranges/lists. Examples:\n"
        "volts=10-60 step 2 suffix \"v\"\n"
        "gauge=15,16,18,23 suffix \" gauge\"",
        "",
        height=120
    )

def parse_generators(gen_text):
    specs = []
    for line in gen_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # pattern: name=START-END step N suffix "txt"
        m = re.match(r".*?=(\d+)\s*-\s*(\d+)\s*step\s*(\d+)\s*suffix\s*\"([^\"]+)\"", line, flags=re.I)
        if m:
            start, end, step, suf = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            for n in range(start, end+1, step):
                specs.append(f"{n}{suf}")
        else:
            # Comma list with suffix: name=15,16,18 suffix " gauge"
            m2 = re.match(r".*?=([\d,\s]+)\s*suffix\s*\"([^\"]+)\"", line, flags=re.I)
            if m2:
                nums = [x.strip() for x in m2.group(1).split(",") if x.strip()]
                suf = m2.group(2)
                for n in nums:
                    specs.append(f"{n}{suf}")
    return specs

# ---------------------- UI: Synonyms ----------------------
with st.expander("🔁 Synonyms (optional)", expanded=False):
    synonyms_text = st.text_area("Pairs: canonical: a|b|c", "whipper snipper: line trimmer|string trimmer", height=100)

def parse_synonyms(txt):
    synmap = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        canon, alts = line.split(":", 1)
        alts = [a.strip().lower() for a in re.split(r"[|,]", alts) if a.strip()]
        synmap[canon.strip().lower()] = alts
    return synmap

def expand_with_synonyms(seed, synmap):
    out = {seed}
    sl = seed.lower().strip()
    for canon, alts in synmap.items():
        if sl == canon or sl in alts:
            out |= set([canon] + alts)
    return sorted(out)

# ---------------------- Seed expansion core ----------------------
def expand_seed(seed: str):
    out_rows = []
    got = set()

    # Merge attributes from manual + external sources
    brands_manual   = _to_list(st.session_state["brands_text"])
    specs_manual    = _to_list(st.session_state["specs_text"])
    features_manual = _to_list(st.session_state["features_text"])
    units_manual    = _to_list(st.session_state["units_text"])

    ext = {"brand": [], "spec": [], "feature": [], "unit": []}
    if mode == "Upload CSV" and uploaded_attrs is not None:
        try:
            ext = load_attributes_from_csv(uploaded_attrs)
        except Exception:
            st.warning("Could not read uploaded CSV. Ensure columns: type,value")
    elif mode == "Google Sheet (CSV URL)" and sheet_url.strip():
        try:
            ext = load_attributes_from_csv(sheet_url.strip())
        except Exception:
            st.warning("Could not load Google Sheet CSV. Check URL/permissions.")

    brands   = sorted(set(brands_manual)   | set(ext["brand"]))
    specs    = sorted(set(specs_manual)    | set(ext["spec"]))
    features = sorted(set(features_manual) | set(ext["feature"]))
    units    = sorted(set(units_manual)    | set(ext["unit"]))

    # Inject generated specs
    gen_specs_list = parse_generators(gen_specs)
    if gen_specs_list:
        specs = sorted(set(specs) | set([s.lower() for s in gen_specs_list]))

    # Synonyms
    synmap = parse_synonyms(synonyms_text)
    seed_variants = expand_with_synonyms(seed, synmap)

    # Templates (if any)
    templates = [t for t in templates_text.splitlines() if t.strip()]

    # 1) Attribute-driven candidates from templates
    if use_attr_mode and templates:
        for seed_variant in seed_variants:
            cand_templates = render_templates(templates, seed_variant, brands, specs, features, limit_per_template=8000)
            for cand in cand_templates:
                suggs = fetch_from_sources(cand, sources, country=country)
                for s in suggs:
                    sl = s.lower().strip()
                    if not sl or sl in got: 
                        continue

                    # Attribute gating
                    hits = 0
                    hits += count_attribute_hits(sl, brands)
                    hits += count_attribute_hits(sl, specs)
                    hits += count_attribute_hits(sl, features)

                    if hits < min_attrs_required:
                        continue
                    if require_number and not has_number(sl):
                        continue
                    if require_unit and units and not has_unit(sl, units):
                        continue

                    # Multi-source validation (optional; stricter)
                    if not validate_keyword(s, country, require_sources=2):
                        continue

                    got.add(sl)
                    out_rows.append({"Keyword": s.strip(), "Seed": seed, "SourceQuery": cand})
                if len(out_rows) >= min_per_seed:
                    break
            if len(out_rows) >= min_per_seed:
                break

    # 2) Attribute combinator (brand/spec/feature mixing) if templates didn't fill enough
    if use_attr_mode and len(out_rows) < min_per_seed:
        for seed_variant in seed_variants:
            attr_candidates = generate_attribute_candidates(seed_variant, brands, specs, features)
            for cand in attr_candidates:
                suggs = fetch_from_sources(cand, sources, country=country)
                for s in suggs:
                    sl = s.lower().strip()
                    if not sl or sl in got:
                        continue

                    hits = 0
                    hits += count_attribute_hits(sl, brands)
                    hits += count_attribute_hits(sl, specs)
                    hits += count_attribute_hits(sl, features)
                    if hits < min_attrs_required:
                        continue
                    if require_number and not has_number(sl):
                        continue
                    if require_unit and units and not has_unit(sl, units):
                        continue
                    if not validate_keyword(s, country, require_sources=2):
                        continue

                    got.add(sl)
                    out_rows.append({"Keyword": s.strip(), "Seed": seed, "SourceQuery": cand})
                if len(out_rows) >= min_per_seed:
                    break
            if len(out_rows) >= min_per_seed:
                break

    # 3) Fallback: broad expansions
    if len(out_rows) < min_per_seed:
        exps = expansions(seed, breadth=breadth)
        for ex in exps:
            suggs = fetch_from_sources(ex, sources, country=country)
            for s in suggs:
                sl = s.lower().strip()
                if not sl or sl in got:
                    continue
                got.add(sl)
                out_rows.append({"Keyword": s.strip(), "Seed": seed, "SourceQuery": ex})
            if len(out_rows) >= min_per_seed:
                break

    # 4) Optional PAA/Related enrichment
    if fetch_paa and len(out_rows) < min_per_seed:
        more = google_related_and_paa(seed)
        for s in more:
            sl = s.lower().strip()
            if not sl or sl in got:
                continue
            got.add(sl)
            out_rows.append({"Keyword": s.strip(), "Seed": seed, "SourceQuery": seed + " (related/paa)"})
            if len(out_rows) >= min_per_seed:
                break

    return pd.DataFrame(out_rows)

def parse_list(s: str):
    if not s.strip():
        return []
    parts = re.split(r"[,\n]+", s.strip())
    return [p.strip().lower() for p in parts if p.strip()]

# ---------------------- Seeds & run ----------------------
st.subheader("Seed Keywords")
tab1, tab2 = st.tabs(["Single seed (Semrush-style explore)","Multiple seeds (batch)"])
with tab1:
    seed_single = st.text_input("Enter a single seed keyword", placeholder="e.g. whipper snipper")
with tab2:
    seeds_multi = st.text_area("Enter multiple seeds (one per line)", height=160, placeholder="e.g.\nwhipper snipper\nnail gun\nimpact driver")

run_single = st.button("🔍 Explore Single Seed")
run_multi = st.button("🚀 Run Batch")

if run_single:
    if not seed_single.strip():
        st.warning("Please enter a seed keyword.")
        st.stop()
    with st.spinner("Fetching suggestions…"):
        df = expand_seed(seed_single.strip())
    if df.empty:
        st.warning("No suggestions found. Try different sources or increase breadth / relax attribute filters.")
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
st.caption("⚠️ Endpoints are unofficial. Use modest rates and consider proxies for larger runs. For large CSV/Sheet lists, start with smaller breadth and loosen filters gradually.")
