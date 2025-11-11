# Earnings Call Sentiment and Stock Returns
Transformer-based NLP pipeline analyzing 77k earnings-call transcripts to link corporate tone with short-term stock returns.

## Overview
This project builds an end-to-end NLP pipeline that analyzes **77k corporate earnings-call transcripts (2010–2025)** to quantify tone and emotion using transformer-based models (**FinBERT + emotion classifier**).  
Developed as an independent research project at **TU Dortmund University**, the system aggregates token-level outputs into interpretable call-level sentiment features and links them with short-term **post-call stock returns** through regression analysis.


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
- `03_emotion_analysis.ipynb`: Applies the emotion classifier to transcripts, explores emotion distributions, and merges with returns.
- `04_sentiment_returns_regression.ipynb`: Regression suite testing FinBERT and emotion features against post-call returns (single-variable, multivariate, etc.).

## Datasets

- `data/transcripts.parquet`: Raw earnings-call transcripts with `symbol`, `year`, `quarter`, and the full transcript text.
- `data/ec_mapping.parquet`: Maps each transcript (`symbol`, `year`, `quarter`) to the earnings-call date `rdq` used to align with market data.
- `data/algoseek_nyse_nasdaq.parquet`: Daily OHLCV data (including `CloseAdjusted`) for NYSE/NASDAQ tickers, used to compute post-call returns.
- `data/additional_ec_data.parquet`: Quarterly fundamentals keyed by ticker (`tic`) and event date (`rdq_x`); useful for control variables.
- `data/var_descriptions.parquet`: Metadata describing the variables in the additional fundamentals dataset.
- `output/sentiment_features*.parquet` / `output/emotion_sentiment_features.parquet`: Feature tables produced by the scoring notebooks (FinBERT and emotion models).

