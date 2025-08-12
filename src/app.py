from flask import Flask, request, render_template, url_for
import pandas as pd
import joblib
import os
import sys

# --- FIX FOR MODULE NOT FOUND ERROR ---
# The previous fix added the project root to the path. A more direct way
# is to use a relative import, which is often more robust within a package.
# We are changing the import statement below.
# ------------------------------------

# Now this import will work correctly because the leading dot '.' tells Python
# to look inside the current package ('src') for the 'data' module.
from .data.preprocess import advanced_feature_engineering


# Initialize the Flask application instance.
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# --- Load Model and Artifacts ---
try:
    # We construct paths relative to this file's location for robustness.
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model = joblib.load(os.path.join(models_dir, 'final_waste_recycling_model.pkl'))
    target_map = joblib.load(os.path.join(models_dir, 'city_target_map.pkl'))
    clusterer = joblib.load(os.path.join(models_dir, 'geo_clusterer.pkl'))
    training_columns = joblib.load(os.path.join(models_dir, 'training_columns.pkl'))
    city_list = sorted([city for city in target_map.keys() if city != '__global_mean__'])
    print("Model and artifacts loaded successfully for the web app.")
except FileNotFoundError as e:
    model = None
    city_list = []
    print(f"FATAL ERROR: Could not load model artifacts. {e}")
    print("Please ensure you have run 'python src/models/train.py' from the project root directory first.")


@app.route('/')
def home():
    """Renders the main input form page ('index.html')."""
    if model is None:
        return "Error: Model not loaded. Please train the model first by running `python src/models/train.py` from your terminal.", 500
    return render_template('index.html', cities=city_list)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form and displays the result."""
    if model is None:
        return render_template('result.html', error="Model is not loaded. Cannot make a prediction.")

    try:
        # Step 1: Collect form data into a DataFrame.
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

        # Step 2: Convert numerical columns from strings to numbers.
        numeric_cols = [
            'Waste Generated (Tons/Day)', 'Recycling Rate (%)', 'Population Density (People/km²)',
            'Municipal Efficiency Score (1-10)', 'Cost of Waste Management (₹/Ton)',
            'Awareness Campaigns Count', 'Landfill Capacity (Tons)', 'Year'
        ]
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Step 3: Preprocess the input using the same pipeline from training.
        processed_df = advanced_feature_engineering(
            input_df.copy(), is_training=False, target_map=target_map, clusterer=clusterer
        )

        # Step 4: Align columns to match the training data format perfectly.
        aligned_df = processed_df.reindex(columns=training_columns, fill_value=0)

        # Step 5: Make the final prediction.
        prediction = model.predict(aligned_df)[0]
        
        # Step 6: Render the result page with the prediction.
        return render_template('result.html', prediction=f"{prediction:.2f}")

    except Exception as e:
        # Handle any unexpected errors during the process.
        print(f"An error occurred during prediction: {e}")
        return render_template('result.html', error="An unexpected error occurred. Please check the input values.")


# This block allows the script to be run directly for local testing.
if __name__ == "__main__":
    # For deployment, the port is often set by the environment.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
