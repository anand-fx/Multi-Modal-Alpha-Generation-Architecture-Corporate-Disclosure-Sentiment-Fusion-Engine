import pandas as pd

def align_bse_timestamps_to_5m(df_bse_scraped):
    """
    Handles Step 4: Normalizes and ceils time strings forward to match intraday charts.
    Assumes incoming dataframe contains a 'timestamp' column from your scraper.
    """
    df_aligned = df_bse_scraped.copy()
    
    # Cast raw string metadata into native datetime objects
    df_aligned['timestamp'] = pd.to_datetime(df_aligned['timestamp'])
    
    # CRITICAL QUANT HYGIENE: Ceil timestamps to the closest future 5-minute interval.
    # An announcement landing at 11:16:02 is forcefully pushed to the 11:20:00 row.
    # This guarantees the model does not access future variables inside past candles.
    df_aligned['target_5m_bar'] = df_aligned['timestamp'].dt.ceil('5min')
    
    return df_aligned