import pandas as pd
import joblib
import sys
import os

# Allow imports from the parent 'src' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocess import advanced_feature_engineering

def make_predictions(input_csv_path, output_csv_path):
    """
    This script defines a reusable prediction pipeline. It loads the trained model
    and all necessary artifacts to make predictions on a new, unseen CSV file.
    This is ideal for generating the final submission file.
    """
    print("--- Starting Prediction Pipeline ---")

    # Step 1: Load the trained model and all transformation artifacts from disk.
    try:
        model = joblib.load('models/final_waste_recycling_model.pkl')
        target_map = joblib.load('models/city_target_map.pkl')
        clusterer = joblib.load('models/geo_clusterer.pkl')
        training_columns = joblib.load('models/training_columns.pkl')
        print("Model and artifacts loaded successfully.")
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run 'train.py' to generate them first.")
        return

    # Step 2: Load the new input data that needs predictions.
    try:
        df_input = pd.read_csv(input_csv_path)
        print(f"Loaded input data from '{input_csv_path}'.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'.")
        return

    # Step 3: Process the input data using the *exact same* pipeline as the training data.
    # We pass the fitted transformers (map, clusterer) to ensure consistency.
    df_processed = advanced_feature_engineering(
        df_input, is_training=False, target_map=target_map, clusterer=clusterer
    )

    # Step 4: Align Columns. This is a critical and robust step.
    # It ensures the prediction data has the exact same columns in the exact same order
    # as the data the model was trained on. Any missing columns are added and filled with 0.
    df_aligned = df_processed.reindex(columns=training_columns, fill_value=0)

    # Step 5: Use the loaded model to make predictions on the aligned data.
    print("Making predictions...")
    predictions = model.predict(df_aligned)

    # Step 6: Save the predictions to a new CSV file.
    # It's good practice to add the prediction as a new column to the original input data.
    df_input['Predicted_Recycling_Rate'] = predictions
    df_input.to_csv(output_csv_path, index=False)
    print(f"Predictions successfully saved to '{output_csv_path}'.")
    print("\n--- Prediction Pipeline Complete ---")

if __name__ == '__main__':
    # This makes the script runnable from the command line.
    # It's configured to generate the 'predictions.csv' file required for the hackathon submission.
    make_predictions('data/raw/Waste_Management_and_Recycling_India.csv', 'predictions.csv')
