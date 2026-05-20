# Multi-Modal-Alpha-Generation-Architecture-Corporate-Disclosure-Sentiment-Fusion-Engine
 A modular, production-grade quantitative finance pipeline that extracts unstructured corporate disclosure data (such as earnings media releases and credit rating rationales), processes the text strings using a deep-learning **FinBERT Transformer Architecture**, and maps directional sentiment indicators onto an intraday 5-minute pricing backbone.


## 🌟 Strategic Project Highlights (For Quant Recruiters)
*   **Zero Lookahead Bias:** Implements strict algorithmic execution hygiene. Non-synchronous textual alerts dropping mid-candle are forcefully mapped forward to the *next available 5-minute candle open boundary* via a temporal ceiling routine.
*   **Transformer Token Windowing:** Employs an overlapping chunk-slicing method to safely process massive regulatory documents without hitting the hard 512-sequence token ceiling of traditional BERT variations.
*   **Production-Grade Architecture:** Decoupled into highly cohesive, single-responsibility modules (`src/`) coordinated by a centralized main orchestrator (`main.py`).

---

## 🏗️ System Architecture & Modular Blueprint

```text
StocksML-Project/
│
├── data/
│   ├── reliance_5m_ohlc.csv         # 100% modeling quality tick-to-5m market pricing
│   └── scraped_bse_data.csv         # Extracted corporate disclosure text database
│
├── src/
│   ├── __init__.py
│   ├── bse_scraper.py               # Robust raw data ingestion layer
│   ├── tokenchunk_finbert.py        # Overlapping token chunking & PyTorch FinBERT engine
│   ├── timemapping.py               # Algorithmic time alignment layer (Anti-Bias)
│   └── multimodal_matrix.py         # Multi-modal technical & text feature fusion layer
│
├── main.py                          # Central system orchestrator
├── requirements.txt                 # Absolute project package dependencies
└── README.md                        # Portfolio documentation


====================================================
✅ PIPELINE EXECUTION COMPLETELY STABLE & RELIABLE  
====================================================
Synchronized Feature Shocks Matrix:
          timestamp        Close  f_text_sentiment
2026-05-20 10:15:00  2804.152310          0.941215   <- (Positive Revenue Shock)
2026-05-20 13:45:00  2799.314502         -0.963140   <- (Negative Credit Rating Downgrade)
====================================================

