# Earnings Call Sentiment and Stock Returns
Transformer-based NLP pipeline analyzing 77k earnings-call transcripts to link corporate tone with short-term stock returns.

## Overview
This project builds an end-to-end NLP pipeline that analyzes **77k corporate earnings-call transcripts (2010–2025)** to quantify tone and emotion using transformer-based models ([FinBERT](https://huggingface.co/ProsusAI/finbert) + [Emotion DistilRoBERTa](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)).  
Developed as an independent research project at **TU Dortmund University**, the system aggregates token-level outputs into interpretable call-level sentiment features and links them with short-term **post-call stock returns** through regression analysis.
**Results** show that tone variables are individually **highly significant** yet they only explain a **small share of return variance**, which is consistent with market efficiency. More details below.

## Structure
```
┌────────────────────────┐        ┌────────────────────────────┐        ┌───────────────────────────┐
│ 1                      │        │ 2                          │        │ 3                         │
│ Data Preprocessing     │ ─────► │ Transformer Inference      │ ─────► │ Feature Engineering       │
│                        │        │                            │        │                           │
│ • Clean + normalize    │        │ • Sentim. & Emotion models │        │ • Aggregate token probs   │
│ • Sentence splitting   │        │ • Batched inference (GPU)  │        │ • Weighted means, entropy │
│ • Chunk to 512 tokens  │        │ • Cache intermediate reps  │        │ • Dispersion, extremes    │
└────────────────────────┘        └────────────────────────────┘        └───────────────────────────┘
                                                                                           │
                                                                                           ▼
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐       ┌────────────────────────────┐        ┌────────────────────────┐
│ 6                          │       │ 5                          │        │ 4                      │
│ Visualization & Analysis   │ ◄──── │ Regression Modeling        │ ◄───── │ Market Data Alignment  │
│                            │       │                            │        │                        │
│ • Correlations, deciles    │       │ • OLS, statsmodels         │        │ • Merge tone + prices  │
│ • Coeff. trends, heatmaps  │       │ • Evaluate R², RMSE        │        │ • Compute 1d/3d/5d     │
│ • Interpret market signal  │       │ • Combined Sentim.+Emotion │        │   forward returns      │
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘       └────────────────────────────┘        └────────────────────────┘
```
**Frameworks**
```
PyTorch · Hugging Face🤗 Transformers · spaCy · pandas · pyarrow · scikit-learn · statsmodels · matplotlib · seaborn
```
**Models**
• [FinBERT (ProsusAI/finbert)](https://huggingface.co/ProsusAI/finbert) – financial sentiment classification (positive / negative / neutral).  
• [Emotion DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base)](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) – 7-way emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise).

## Results Summary
Regression analyses were conducted over 62k matched earnings-call transcripts and stock-return observations to evaluate how linguistic tone relates to short-horizon market reactions.  
Three models were tested:
- **FinBERT Sentiment**, 
- **DistilRoBERTa Emotion tone**, 
- **Combined model** integrating both feature sets.

### Key Insights
**Directionally consistent effects**  
  Positive tone and emotional valence are associated with **higher post-call returns**, while negative sentiment and emotions (e.g., sadness, fear) correlate with **lower returns**.

**Short-term concentration** 
  Predictive strength peaks at the **1-day horizon** and decays over 3–5 days, indicating that markets **rapidly incorporate tone information** following earnings calls.

**Complementary signals**  
  The **combined model outperforms** sentiment or emotion features alone, suggesting that emotional nuance adds incremental explanatory power to finance-specific sentiment.

**Statistical strength, modest economic impact**  
  Although coefficients are **highly significant** (t-stats >15 across tone variables), overall explanatory power remains **small (R² ≤ 0.013)**. This is consistent with prior literature on financial-text analytics.

**Interpretation**  
  Results confirm that **corporate communication tone carries short-lived yet measurable predictive information**, reinforcing the role of NLP-derived sentiment and emotion features as **auxiliary signals** in quantitative trading and event-driven models.

### Research Scope & Implications
This analysis isolates the linguistic component of market reactions, focusing on *tone–return causality* rather than price levels or the basics.

Future work could extend this by:
- Incorporating **firm and sector fixed effects** to control for baseline performance.  
- Modeling **interaction terms** between tone and earnings surprises.  
- Testing **temporal dynamics** (rolling regressions, expanding windows) for stability.  
- Exploring **multi-modal features** (audio tone, speaker sentiment, Q&A sections).

---

## Environment setup
```bash
# Clone the repository
git clone https://github.com/aryahassibi/Earnings-Call-Sentiment-and-Stock-Returns.git
cd Earnings-Call-Sentiment-and-Stock-Returns

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
# update reqs with pip freeze > requirements.txt
```
Optionally, install the spaCy language model 
```bash
python -m spacy download en_core_web_sm
```

and register a Jupyter kernel for the virtual environment to appear in the Jupyter UI
```
# Register Jupyter kernel
python -m ipykernel install --user --name earnings-call-sentiment --display-name "Earnings Call Sentiment"
```

## Utilities

Run `python utils/analyze_parquet.py <parquet_file>` to generate a JSON profile and Markdown summary for any dataset. Use `--help` for options (sampling, alternate text columns, pandas vs. pyarrow engine).

`process_var_description_docx.py` extracts variable descriptions from the docx file and saves them as a Parquet file.


## Notebooks
- `00_data_overview.ipynb`: dataset exploration, health checks, and transcript statistics.
- `01_sentiment_scoring.ipynb`: production-ready sentiment feature pipeline with caching.
- `01v2_sentiment_scoring.ipynb`: Runs FinBERT ( or `yiyanghkust/finbert-tone`) on transcripts; outputs call-level sentiment features (including prepared/Q&A and speaker-role deltas).
- `02_returns_regression.ipynb`: merge sentiment features with market data and fit baseline regressions.
- `03_emotion_analysis.ipynb`: Applies the DistilRoBERTa emotion classifier to transcripts, explores emotion distributions, and merges with returns.
- `04_sentiment_returns_regression.ipynb`: Regression suite testing FinBERT and DistilRoBERTa features against post-call returns (single-variable, multivariate, etc.).

## Datasets

- `data/transcripts.parquet`: Raw earnings-call transcripts with `symbol`, `year`, `quarter`, and the full transcript text.
- `data/ec_mapping.parquet`: Maps each transcript (`symbol`, `year`, `quarter`) to the earnings-call date `rdq` used to align with market data.
- `data/algoseek_nyse_nasdaq.parquet`: Daily OHLCV data (including `CloseAdjusted`) for NYSE/NASDAQ tickers, used to compute post-call returns.
- `data/additional_ec_data.parquet`: Quarterly fundamentals keyed by ticker (`tic`) and event date (`rdq_x`); useful for control variables.
- `data/var_descriptions.parquet`: Metadata describing the variables in the additional fundamentals dataset.
- `output/sentiment_features*.parquet` / `output/emotion_sentiment_features.parquet`: Feature tables produced by the scoring notebooks (FinBERT and DistilRoBERTa emotion models).

