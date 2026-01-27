import os
import sys
import time

# Import functions from our modular scripts
from preprocess_data import preprocess
from sales_forecasting import train_model
from predict_future import generate_forecast

def main():
    print("====================================================")
    print("--- WELCOME TO THE ULTIMATE SALES FORECASTING SYSTEM ---")
    print("====================================================")
    
    start_time = time.time()

    # Paths
    PROCESSED_FILE = os.path.join('data', 'processed_daily_sales.csv')
    MODEL_PATH = os.path.join('models', 'sales_model.joblib')
    RAW_DATA = os.path.join('raw_data', 'Online Retail.xlsx')

    # Check for force flag
    force_refresh = "--refresh" in sys.argv

    # 1. PREPROCESSING
    if not os.path.exists(PROCESSED_FILE) or force_refresh:
        print("\nStep 1: Raw Data detected. Starting Preprocessing...")
        preprocess()
    else:
        print("\nStep 1: Preprocessed data already exists. Skipping...")

    # 2. TRAINING
    if not os.path.exists(MODEL_PATH) or force_refresh:
        print("\nStep 2: Training the Machine Learning Model...")
        train_model()
    else:
        print("\nStep 2: Trained model already exists. Skipping retrain...")

    # 3. PREDICTION
    print("\nStep 3: Generating Final 30-Day Forecast...")
    generate_forecast()

    end_time = time.time()
    print("\n====================================================")
    print(f"DONE! MISSION COMPLETE! Total Time: {end_time - start_time:.2f} seconds.")
    print("7 ANALYTICAL FORECAST GRAPHS GENERATED!")
    print("Check the 'plots/' folder for files 1 to 7.")
    print("====================================================")
    print("\nTIP: Run 'python main.py --refresh' if your Excel data changes!")

if __name__ == "__main__":
    main()
