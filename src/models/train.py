import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import sys
import os

# --- FIX FOR MODULE NOT FOUND ERROR ---
# This is a critical step for making a modular Python project runnable.
# 1. We get the directory of the current file (`train.py`).
# 2. We go up two levels to get to the project's root directory (`pwskills_hackathon_waste`).
# 3. We add this root directory to the VERY BEGINNING of Python's path.
# This ensures that when we say `from src.data...`, Python knows where to find the `src` folder.
#
# DEBUGGING CHECKLIST IF THIS ERROR PERSISTS:
#   1. Are you running this script from the project's ROOT folder (`pwskills_hackathon_waste`)?
#   2. Is your virtual environment (`venv`) active?
#   3. Do you have empty `__init__.py` files in both the `src/` and `src/data/` folders?

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------

# Now this import will work correctly
from src.data.preprocess import advanced_feature_engineering

def train_and_save_model():
    """
    This script serves as the main training pipeline. It automates all steps
    from loading raw data to saving a final, tuned model.
    """
    print("--- Starting Model Training Pipeline ---")

    # Step 1: Load the raw dataset from the specified path.
    try:
        df_raw = pd.read_csv('data/raw/Waste_Management_and_Recycling_India.csv')
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print("Error: Raw data file not found. Make sure 'Waste_Management_and_Recycling_India.csv' is in the 'data/raw/' directory.")
        return

    # Step 2: Apply the preprocessing pipeline defined in `preprocess.py`.
    print("Processing data and engineering features...")
    df_processed, target_map, clusterer = advanced_feature_engineering(df_raw, is_training=True)
    
    # Separate features (X) from the target variable (y).
    X = df_processed.drop('Recycling Rate (%)', axis=1)
    y = df_processed['Recycling Rate (%)']
    
    # Store the column names and order from the training data.
    training_columns = X.columns.tolist()

    # Step 3: Use RandomizedSearchCV to find the best hyperparameters for the XGBoost model.
    print("Tuning XGBoost model...")
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=30, cv=5, verbose=1, n_jobs=-1,
        scoring='neg_root_mean_squared_error', random_state=42
    )
    random_search.fit(X, y)

    print(f"Best parameters found: {random_search.best_params_}")
    final_model = random_search.best_estimator_

    # Step 4: Save all the necessary "artifacts" for deployment and prediction.
    print("Saving final model and transformation artifacts...")
    joblib.dump(final_model, 'models/final_waste_recycling_model.pkl')
    joblib.dump(target_map, 'models/city_target_map.pkl')
    joblib.dump(clusterer, 'models/geo_clusterer.pkl')
    joblib.dump(training_columns, 'models/training_columns.pkl')

    print("\n--- Model Training Pipeline Complete ---")
    print("All necessary artifacts have been saved to the 'models/' directory.")

if __name__ == '__main__':
    # This block allows the script to be run directly from the command line.
    train_and_save_model()
