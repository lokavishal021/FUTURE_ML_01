import pandas as pd
import os
import time

def preprocess():
    print("--- Starting Heavy Data Preprocessing ---")
    start_time = time.time()

    # 1. Path Configuration
    RAW_FILE = os.path.join('raw_data', 'Online Retail.xlsx')
    PROCESSED_FOLDER = 'data'
    OUTPUT_FILE = os.path.join(PROCESSED_FOLDER, 'processed_daily_sales.csv')

    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    # 2. Heavy Load (The bottle-neck)
    print(f"--- Reading raw Excel file from: {RAW_FILE} ---")
    print("--- (This might take 30-60 seconds...) ---")
    df = pd.read_excel(RAW_FILE)

    # 3. Cleaning & Transformation
    print("--- Cleaning and Aggregating ---")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # Filter out bad records
    clean_df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    clean_df = clean_df.dropna(subset=['CustomerID']) 

    # Aggregation to Daily Level
    daily_sales = clean_df.groupby(clean_df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
    daily_sales.columns = ['Date', 'Sales']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date').set_index('Date')

    # Clip outliers once here (using the 5.0 IQR strategy we agreed on)
    Q1 = daily_sales['Sales'].quantile(0.25)
    Q3 = daily_sales['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    upper_cap = Q3 + 5.0 * IQR
    daily_sales['Sales_clipped'] = daily_sales['Sales'].clip(upper=upper_cap)

    # Fill timeline gaps
    daily_sales = daily_sales.asfreq('D').fillna(0)

    # 4. Save to Lightweight CSV
    print(f"--- Saving refined data to: {OUTPUT_FILE} ---")
    daily_sales.to_csv(OUTPUT_FILE)
    
    end_time = time.time()
    print(f"--- DONE! Total preprocessing time: {end_time - start_time:.2f} seconds. ---")
    print("Now your ML script will run almost instantly!")

if __name__ == "__main__":
    preprocess()
