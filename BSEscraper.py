import pandas as pd
import os

def scrape_bse_announcements(period="D"):
    """
    Looks for a local CSV file containing scraped announcements.
    Falls back to a simulation if no file is present.
    """
    local_file = "scraped_bse_data.csv"
    
    # Check if you have saved your scraped data locally
    if os.path.exists(local_file):
        print(f"   📊 Local file found! Loading raw data from {local_file}...")
        try:
            df = pd.read_csv(local_file)
            
            # Standardize column names to match what main.py expects
            if 'timestamp' in df.columns and 'raw_text_extracted' in df.columns:
                return df[['timestamp', 'raw_text_extracted']]
            else:
                raise ValueError("CSV must contain 'timestamp' and 'raw_text_extracted' columns.")
        except Exception as e:
            print(f"   ⚠️ Error reading local CSV: {e}")
            
    # Hardened Fallback Routine (if no local file exists)
    print("   ℹ️ No local scraped_bse_data.csv found. Using sample matrix...")
    return pd.DataFrame({
        'timestamp': ['2026-05-20 10:11:45', '2026-05-20 13:42:10'],
        'raw_text_extracted': [
            "Reliance Industries reports stellar revenue growth in quarterly results. Profits exceed street expectations.",
            "Credit rating outlook revised to negative due to increasing capital expenditure debt pressures."
        ]
    })