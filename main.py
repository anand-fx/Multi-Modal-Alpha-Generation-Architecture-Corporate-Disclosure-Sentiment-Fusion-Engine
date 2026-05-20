import pandas as pd
import numpy as np

# Import the functional logic from your modular project files
from BSEscraper import scrape_bse_announcements
from timemapping import align_bse_timestamps_to_5m
from multimodel_feature_matrix import build_multimodal_training_matrix

# Note: Using dynamic import or handling initialization inside the main block 
# ensures your GPU memory allocation behaves well across modular files.
from tokenchunking_FINbert import FinBERTInferenceEngine


def run_pipeline():
    print("=== [STAGES 1-5]: STARTING QUANT DATA PIPELINE ===")
    
    # -------------------------------------------------------------
    # STAGE 1: Scrape text payload from exchange
    # -------------------------------------------------------------
    print("\n[Stage 1] Fetching regulatory feeds from BSE space...")
    # This calls your scraping function. If you are mock-testing right now,
    # it returns a clean DataFrame with ['timestamp', 'raw_text_extracted']
    df_raw_scraped = scrape_bse_announcements()
    print(f"-> Successfully collected {len(df_raw_scraped)} text documents.")
    
    # -------------------------------------------------------------
    # STAGES 2 & 3: Run Vector Weights & Document Chunk Inference
    # -------------------------------------------------------------
    print("\n[Stage 2 & 3] Loading FinBERT model & running vector inference...")
    inference_worker = FinBERTInferenceEngine()
    
    # Compute continuous sentiment score (-1 to +1) for every release document
    df_raw_scraped['sentiment_score'] = [
        inference_worker.generate_document_sentiment(text) 
        for text in df_raw_scraped['raw_text_extracted']
    ]
    print("-> Sentiment analysis mapping complete.")
    
    # -------------------------------------------------------------
    # STAGE 4: Intraday Time Alignment (No Lookahead Bias)
    # -------------------------------------------------------------
    print("\n[Stage 4] Pushing publication metrics to target 5m time bars...")
    df_text_features = align_bse_timestamps_to_5m(df_raw_scraped)
    print("-> Continuous time boundaries aligned safely via ceil('5min').")
    
    # -------------------------------------------------------------
    # STAGE 5: Multi-Modal Training Matrix Consolidation
    # -------------------------------------------------------------
    print("\n[Stage 5] Fusing technical price backbone with text metrics...")
    
    # Define paths to your high-fidelity tick data converted to 5m bars
    ohlc_5m_path = "reliance_5m_ohlc.csv"
    
    # Quick fallback verification to create a mock market file if you don't have it in directory
    try:
        pd.read_csv(ohlc_5m_path)
    except FileNotFoundError:
        print(f"   ℹ️ {ohlc_5m_path} not found. Generating simulated price spine for test run...")
        time_backbone = pd.date_range(start="2026-05-20 09:15:00", end="2026-05-20 15:30:00", freq="5min")
        pd.DataFrame({
            'timestamp': time_backbone,
            'Open': np.random.uniform(2800, 2810, len(time_backbone)),
            'High': np.random.uniform(2810, 2820, len(time_backbone)),
            'Low': np.random.uniform(2790, 2800, len(time_backbone)),
            'Close': np.random.uniform(2800, 2810, len(time_backbone))
        }).to_csv(ohlc_5m_path, index=False)
    
    # Process final fusion
    final_dataset = build_multimodal_training_matrix(ohlc_5m_path, df_text_features)
    
    # Verify records where news events hit the market spine
    active_announcements = final_dataset[final_dataset['f_text_sentiment'] != 0.0]
    
    print("\n=== PIPELINE PROCESSING SUCCESSFUL ===")
    print("Matrix Sample (Non-zero sentiment slots):")
    if not active_announcements.empty:
        print(active_announcements[['timestamp', 'Close', 'f_text_sentiment']].to_string(index=False))
    else:
        print(final_dataset[['timestamp', 'Close', 'f_text_sentiment']].head(10).to_string(index=False))
        
    # Save the consolidated matrix out for your downstream ML model
    final_dataset.to_csv("features_multimodal_matrix.csv", index=False)
    print("\nSaved ready feature matrix to: features_multimodal_matrix.csv")


if __name__ == "__main__":
    run_pipeline()