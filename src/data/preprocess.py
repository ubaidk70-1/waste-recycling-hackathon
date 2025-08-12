import pandas as pd
from sklearn.cluster import KMeans
import joblib

def advanced_feature_engineering(df, is_training=True, target_map=None, clusterer=None):
    """
    Performs advanced data cleaning and feature engineering.

    This modular function encapsulates all preprocessing logic, ensuring that
    both the training and prediction pipelines use the exact same transformations.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        is_training (bool): Flag to indicate if the function is being used for training.
                            If True, it will create and fit new transformers (maps, clusterers).
                            If False, it will use the provided, pre-fitted transformers.
        target_map (dict, optional): A pre-fitted dictionary for target encoding cities.
                                     Required if is_training is False.
        clusterer (KMeans, optional): A pre-fitted KMeans object for geo-clustering.
                                      Required if is_training is False.
    Returns:
        pd.DataFrame: The fully processed and feature-engineered dataframe.
        dict: The fitted target encoding map (only if is_training).
        KMeans: The fitted KMeans clusterer (only if is_training).
    """
    # --- 1. Initial Cleanup ---
    # The 'Landfill Name' is a unique identifier. It provides no generalizable predictive
    # power and would cause the model to overfit. We drop it safely using errors='ignore'.
    df = df.drop(['Landfill Name'], axis=1, errors='ignore')

    # --- 2. Geospatial Feature Engineering ---
    # Convert the 'Lat, Long' string into two separate numerical columns.
    lat_long = df['Landfill Location (Lat, Long)'].str.split(', ', expand=True)
    df['Latitude'] = pd.to_numeric(lat_long.get(0), errors='coerce')
    df['Longitude'] = pd.to_numeric(lat_long.get(1), errors='coerce')
    
    # Handle any potential missing coordinates by filling them with a central point for India.
    # This ensures the clustering algorithm can run without errors.
    df['Latitude'].fillna(20.5937, inplace=True)
    df['Longitude'].fillna(78.9629, inplace=True)

    # Use KMeans clustering to group cities into geographical regions.
    # This converts two continuous features (lat, long) into one powerful categorical feature.
    if is_training:
        # If training, create and fit a new KMeans model.
        kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
        df['Geo_Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
        clusterer = kmeans  # Store the fitted model to be saved later.
    else:
        # If predicting, use the pre-fitted clusterer to ensure new data is
        # assigned to the same regions defined during training.
        if clusterer is None:
            raise ValueError("A fitted 'clusterer' must be provided when not in training mode.")
        df['Geo_Cluster'] = clusterer.predict(df[['Latitude', 'Longitude']])

    # --- 3. Contextual Feature Engineering ---
    # Create new features (ratios) that capture more nuanced relationships in the data.
    # These often have more predictive power than the raw features alone.
    df['Efficiency_Cost_Index'] = df['Municipal Efficiency Score (1-10)'] / (df['Cost of Waste Management (₹/Ton)'] + 1)
    df['Campaign_Effectiveness_Ratio'] = df['Awareness Campaigns Count'] / (df['Population Density (People/km²)'] + 1)
    df['Capacity_vs_Generation'] = df['Landfill Capacity (Tons)'] / (df['Waste Generated (Tons/Day)'] + 1)

    # --- 4. Advanced Categorical Encoding (Target Encoding) ---
    # 'City/District' has too many unique values for one-hot encoding. Target encoding
    # replaces each city with its average recycling rate, a very powerful predictor.
    if is_training:
        # Create the city-to-average-rate mapping from the training data.
        target_map = df.groupby('City/District')['Recycling Rate (%)'].mean().to_dict()
        # Store the global mean to handle unseen cities during prediction.
        global_mean = df['Recycling Rate (%)'].mean()
        target_map['__global_mean__'] = global_mean
    
    if target_map is None:
        raise ValueError("A 'target_map' must be provided when not in training mode.")
        
    # Apply the map. Any new cities not seen during training will get the global average rate.
    df['City_Target_Encoded'] = df['City/District'].map(target_map).fillna(target_map['__global_mean__'])

    # --- 5. Standard Categorical Encoding (One-Hot) ---
    # Use one-hot encoding for the remaining low-cardinality categorical features.
    # `dtype=int` ensures the new columns are 1s and 0s, not booleans.
    df = pd.get_dummies(
        df,
        columns=['Waste Type', 'Disposal Method', 'Geo_Cluster'],
        drop_first=True,
        dtype=int
    )

    # --- 6. Final Column Cleanup ---
    # Drop the original columns that have been engineered or encoded to avoid data redundancy
    # and the "dummy variable trap".
    df = df.drop(['City/District', 'Landfill Location (Lat, Long)', 'Latitude', 'Longitude'], axis=1)
    
    # Return the processed dataframe and the fitted transformers (if in training mode).
    if is_training:
        return df, target_map, clusterer
    else:
        return df
