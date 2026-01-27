import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

warnings.filterwarnings('ignore')
sns.set(style="whitegrid", palette="muted")

def build_advanced_features(df, target_col='Sales_clipped'):
    data = df.copy()
    # Time features
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    data['is_december'] = (data['month'] == 12).astype(int)
    
    # Festive Feature: Days until Christmas
    data['days_to_christmas'] = data.index.map(lambda x: (pd.Timestamp(year=x.year, month=12, day=25) - x).days)
    data['days_to_christmas'] = data['days_to_christmas'].apply(lambda x: x if 0 <= x <= 30 else 31)

    # Cyclical Seasonality Encoding
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    data['day_sin'] = np.sin(2 * np.pi * data['dayofweek']/7)
    data['day_cos'] = np.cos(2 * np.pi * data['dayofweek']/7)

    # Advanced Lags on the target column
    for lag in [1, 7, 14, 21, 30]:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Differential Lags
    data['diff_1_7'] = data['lag_1'] - data['lag_7']
    
    # Rolling Statistics
    data['rolling_mean_7'] = data[target_col].shift(1).rolling(window=7).mean()
    data['rolling_std_7'] = data[target_col].shift(1).rolling(window=7).std()
    data['rolling_mean_30'] = data[target_col].shift(1).rolling(window=30).mean()
    
    return data.dropna()

def train_model():
    print("--- [1/2] Loading Preprocessed Data for Training ---")
    PROCESSED_FILE = os.path.join('data', 'processed_daily_sales.csv')

    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError("Processed data not found! Please run preprocessing first.")

    daily_sales = pd.read_csv(PROCESSED_FILE)
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date').set_index('Date')

    print(f"--- Loaded {len(daily_sales)} days of sales records. ---")

    print("--- [2/2] Training XGBoost Model ---")
    model_df = build_advanced_features(daily_sales, target_col='Sales_clipped')

    X = model_df.drop(['Sales', 'Sales_clipped'], axis=1)
    y = model_df['Sales_clipped']
    y_real = model_df['Sales']

    # Last 30 days for hold-out validation
    split_idx = len(X) - 30
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_test_real = y_real.iloc[split_idx:]

    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(xgb, param_grid, n_iter=20, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"--- Best Parameters Found: {random_search.best_params_} ---")

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    MODEL_PATH = os.path.join('models', 'sales_model.joblib')
    joblib.dump(best_model, MODEL_PATH)
    print(f"--- Model saved successfully at: {MODEL_PATH} ---")

    # Final Evaluation Plot
    y_pred = best_model.predict(X_test)
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_real.index, y_test_real.values, label='Actual Data (Real)', marker='o', alpha=0.7)
    plt.plot(y_test_real.index, y_pred, label='ML Forecast', color='red', linestyle='--', marker='x')
    plt.title('Validation Strategy: 30-Day Blind Back-Test', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', shadow=True)
    plt.tight_layout()
    plt.savefig('plots/accuracy_test.png', dpi=300)
    plt.close()

    # SAVE VALIDATION DATA FOR DASHBOARD
    validation_df = pd.DataFrame({
        'Date': y_test_real.index,
        'Actual': y_test_real.values,
        'Forecast': y_pred
    })
    validation_df.to_csv('data/validation_results.csv', index=False)
    
    return best_model

if __name__ == "__main__":
    train_model()
