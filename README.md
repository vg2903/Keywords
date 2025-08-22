# 🔎 Keyword Explorer (Streamlit)

Two free Streamlit apps for keyword research:

1. **`app_semrush_like.py`** – A Semrush-style keyword explorer
   - Multi-source (Google, Bing, YouTube, Amazon)
   - Expansion (A–Z, 0–9, prefix/suffix)
   - Filters: word count, include/exclude, regex
   - Intent detection (Informational / Commercial / Transactional)
   - Clustering (SentenceTransformer embeddings or TF-IDF)
   - CSV export

2. **`app.py`** – Simple long-tail keyword finder
   - Google autocomplete only
   - Long-tail filtering (≥3 words)
   - Intent classification
   - Clustering

---

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/keyword-explorer-streamlit.git
cd keyword-explorer-streamlit
pip install -r requirements.txt
streamlit run app_semrush_like.py
