import pandas as pd
import numpy as np

def build_multimodal_training_matrix(ohlc_5m_csv_path, df_processed_sentiment):
    """
    Handles Step 5: Merges technical intervals with text parameters cleanly.
    """
    # 1. Ingest local asset data file layout
    df_ohlc = pd.read_csv(ohlc_5m_csv_path)
    df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'])
    df_ohlc = df_ohlc.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Group overlapping filings mapped to the same 5-minute row
    # If multiple updates drop inside the same 5m bar, calculate their collective mean sentiment
    df_text_grouped = df_processed_sentiment.groupby('target_5m_bar')['sentiment_score'].mean().reset_index()
    df_text_grouped = df_text_grouped.rename(columns={'target_5m_bar': 'timestamp', 'sentiment_score': 'f_text_sentiment'})
    
    # 3. Structural Fusion: Join text variables directly onto the price backbone
    df_final_matrix = pd.merge(df_ohlc, df_text_grouped, on='timestamp', how='left')
    
    # 4. Data Patching: Set missing interval records to 0.0 (Neutral)
    # Text disclosures drop intermittently. The rows between files must pass a flat neutral state.
    df_final_matrix['f_text_sentiment'] = df_final_matrix['f_text_sentiment'].fillna(0.0)
    
    return df_final_matrix