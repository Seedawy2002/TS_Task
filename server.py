from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Helper function to preprocess the dataset
def preprocess_data(df):
    # Ensure 'timestamp' and 'value' columns exist
    if 'timestamp' not in df.columns:
        raise KeyError("'timestamp' column is missing from the input data")
    
    if 'value' not in df.columns:
        raise KeyError("'value' column is missing from the input data")

    # Convert 'timestamp' to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Apply forward fill first
    df.ffill(inplace=True)

    # Apply backfill to handle remaining NaNs at the start
    df.bfill(inplace=True)
    
    # If NaNs persist, drop them if needed
    df.dropna(inplace=True)
    
    # Feature Engineering: 
    # 1. Add lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)

    # Backfill for initial NaNs in lag features
    df[['lag_1', 'lag_2', 'lag_3']] = df[['lag_1', 'lag_2', 'lag_3']].bfill()

    if 'anomaly' in df.columns:
        # Forward fill the 'value' column only where 'anomaly' is 1
        df.loc[df['anomaly'] == True, 'value'] = pd.NA  # Mark as NaN where anomaly is detected
        df['value'] = df['value'].ffill()  # Forward fill to replace incorrect values
        
        # Drop the 'anomaly' column after making the changes
        df = df.drop('anomaly', axis=1)
    
    # 2. Add Rate of Change (Derivatives)
    df['rate_of_change'] = df['value'].diff()
    df['rate_of_change_2'] = df['rate_of_change'].diff()

    # Fill NaNs from differencing
    df[['rate_of_change', 'rate_of_change_2']] = df[['rate_of_change', 'rate_of_change_2']].bfill()

    # 3. Add Rolling Window Statistics (Moving Average, Std Dev, Min, Max)
    df['rolling_mean_5'] = df['value'].rolling(window=5).mean()
    df['rolling_std_5'] = df['value'].rolling(window=5).std()
    df['rolling_min_5'] = df['value'].rolling(window=5).min()
    df['rolling_max_5'] = df['value'].rolling(window=5).max()

    # Fill NaNs from rolling windows
    df[['rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5']] = df[['rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5']].bfill()

    # 4. Add Exponential Moving Average (EMA)
    df['ema_5'] = df['value'].ewm(span=5, adjust=False).mean()

    # Fill NaNs from EMA
    # Backfill for 'ema_5' column without inplace=True
    df['ema_5'] = df['ema_5'].bfill()

    # Apply forward fill first
    df.ffill(inplace=True)
    
    # 5. Add Outlier Detection (Z-score)
    df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    df['is_anomaly'] = (df['z_score'].abs() > 3).astype(int)  # Mark anomalies where Z-score exceeds threshold

    # 6. Add Time-Based Features (Hour, Day, Month)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    # 7. Add Cyclic Feature Encoding (Sin/Cos Transforms for Time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # If NaNs persist, drop them if needed
    df.dropna(inplace=True)
    
    # Scaling the features
    scaler = StandardScaler()
    feature_columns = ['value', 'lag_1', 'lag_2', 'lag_3', 
                       'rate_of_change', 'rate_of_change_2', 
                       'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5', 
                       'ema_5', 'z_score', 'is_anomaly',
                       'hour', 'day_of_week', 'day_of_month', 'month', 
                       'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']

    # Apply scaling only to the selected columns
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    #print(df)

    return df

# Function to load a saved model
def load_model(file_name):
    model_load_path = f"saved_models/{file_name}.pkl"
    if os.path.exists(model_load_path):
        model = joblib.load(model_load_path)
        print(f"Model loaded from {model_load_path}")
        return model
    else:
        print(f"No model found at {model_load_path}")
        return None
    
app = Flask(__name__)

# Route that accepts JSON input and returns processed JSON output
@app.route('/process', methods=['POST'])
def process_json():
    # Get the JSON from the request
    input_data = request.get_json()

    # Extract dataset ID and values from the request
    dataset_id = input_data.get('dataset_id')
    values = input_data.get('values')

    # Convert values into a DataFrame (assuming values are in the format [{'timestamp': ..., 'value': ..., 'anomaly': ...}])
    df = pd.DataFrame(values)

    # Check if 'timestamp' and 'value' columns exist before preprocessing
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        raise KeyError("The input data must contain 'timestamp' and 'value' columns.")
    
    # Preprocess the test data before feeding it into the model
    df = preprocess_data(df)

    # Define all feature columns for prediction (use all generated features)
    feature_columns = ['lag_1','lag_2','lag_3','rate_of_change', 'rate_of_change_2', 
           'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5', 
           'ema_5', 'z_score',
           'hour', 'day_of_week', 'day_of_month', 'month', 
           'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    
    # Include 'anomaly' if it exists in the DataFrame
    if 'anomaly' in df.columns:
        feature_columns.append('anomaly')

    # Ensure the correct features are used for prediction
    # Check if the selected feature columns exist in the DataFrame after preprocessing
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"The following required feature columns are missing after preprocessing: {missing_features}")
    
    # Extract the feature columns
    features = df[feature_columns]

    # Debugging: Print the features passed for prediction (can be removed later)
    # print("Features passed for prediction:")
    # print(features)

    # Load the correct model based on dataset_id
    model_path = rf'C:\Users\Mariam Magdy\OneDrive\Desktop\ts_task\models\train_{dataset_id}.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file for dataset_id {dataset_id} not found at {model_path}")
    
    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Return the processed data as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)