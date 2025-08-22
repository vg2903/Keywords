# 🧠 Semrush-like Keyword Explorer (Free, Pro Edition)

A free, Streamlit-based keyword explorer inspired by Semrush.  
Designed for **SEO automation, programmatic keyword research, and long-tail generation**.

---

## 🚀 Features
- **Autocomplete Sources**: Google, Bing, YouTube, Amazon  
- **Country Selector**: Region-specific suggestions (e.g., AU, US, IN)  
- **Attribute-driven Refinement**:  
  - Brands, Specs, Features, Units  
  - Preset packs (Power Tools, Garden Tools)  
  - CSV/Google Sheet import  
  - Synonyms (e.g., *whipper snipper* → *line trimmer*, *string trimmer*)  
- **Templates Engine**: `{brand} {spec} {seed}`, `{spec} {seed} for {feature}`, etc.  
- **Numeric Range DSL**: `volts=10-60 step 2 suffix "v"`, `gauge=15,16,18,23 suffix " gauge"`  
- **AI Assist (OpenAI)**:  
  - Generate attribute packs automatically  
  - Extract attributes from SERP snippets  
- **Wikidata Enrichment**: Auto-discover brands from Wikidata  
- **Validation**: Keep only phrases that appear across 2+ autocomplete sources  
- **Scoring**: Heuristic (multi-source, attributes, units, numbers) + optional Google Trends  
- **Clustering & Intent Detection**: Informational / Commercial / Transactional  
- **Export**: Download results as CSV  

---

## 🛠 Setup

### 1. Clone repo
```bash
git clone https://github.com/your-username/keyword-explorer.git
cd keyword-explorer
