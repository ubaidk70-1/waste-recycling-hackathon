# Mini-Hackathon: Waste Management & Recycling Prediction

This project is a submission for the **PWSkills "Waste Management and Recycling in Indian Cities" mini-hackathon**.  
The goal is to **predict the recycling rate (%)** based on various city-specific attributes using machine learning.

The project follows a structured **data science workflow** — from data exploration and advanced feature engineering to model training, optimization, and deployment via a **Flask web application**.

---

## Final Model & Key Findings

The final model chosen for this project is a **Tuned XGBoost Regressor**.

**Model Development Process:**
- **Baseline:** Linear Regression model with RMSE ≈ 16.5 → too high, indicating complexity in data.
- **Advanced Models Tested:** Random Forest and default XGBoost.
- **Key Insight:** Even after hyperparameter tuning, models couldn’t significantly outperform the baseline — suggesting a **predictive limit of the available features**.
- **Final Decision:** Tuned XGBoost was selected due to **theoretical robustness** and **feature importance analysis** capabilities.
- **Conclusion:** To reduce error further, more features are needed (e.g., local policy data, economic indicators).

---




## How to Run This Project

### 1. Setup Environment
Clone the repository and navigate into the project directory.  
It is recommended to use a Python virtual environment.


#### Clone the repository
```
git clone <your-repo-url>
cd pwskills_hackathon_waste
```
#### Create and activate a virtual environment
```
python -m venv venv
```

#### On Windows:
```
venv\Scripts\activate
```
### On macOS/Linux:
```
source venv/bin/activate
```
### Install all dependencies

```
pip install -r requirements.txt
```
### 2️. Train the Model
Run the training script to perform all data preprocessing, model tuning, and save the final model along with all transformation artifacts into the `models/` directory.

```
python src/models/train.py
```

### 3️. Run the Web Application
Once the model is trained, you can start the Flask web application.

#### On Windows:
```
waitress-serve --host 127.0.0.1 --port=8000 src.app:app
```

#### On macOS/Linux:

```
gunicorn src.app:app
```
Open your browser and navigate to:
http://127.0.0.1:8000

### 4️. Generate Submission Predictions
To generate the `predictions.csv` file required for the submission, run the prediction script.
This will load the trained model and predict on the raw dataset.

```
python src/models/predict.py
```
The `predictions.csv` file will be created in the root directory of your project.
