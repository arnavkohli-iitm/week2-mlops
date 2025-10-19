import joblib
import sys
import pandas as pd

# 1. Define paths
MODEL_PATH = 'artifacts/model.joblib'

def predict(features):
    """
    Loads the model and makes a prediction.
    """
    try:
        # 2. Load the model
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: Model file not found.")
        print("Please run 'dvc pull' to download the model from GCS.")
        return

    # 3. Format features
    # We expect 4 features from the command line
    try:
        # Convert string inputs to floats
        feature_values = [float(f) for f in features]
        
        # Create a DataFrame with the correct feature names 
        # (Assuming standard iris dataset order: sepal_len, sepal_wid, petal_len, petal_wid)
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        data_df = pd.DataFrame([feature_values], columns=column_names)

    except Exception as e:
        print(f"Error: Invalid input features. Expected 4 numbers. Got: {features}")
        print(e)
        return

    # 4. Make prediction
    prediction = model.predict(data_df)
    print(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        print("Example: python predict.py 5.1 3.5 1.4 0.2")
    else:
        # Pass all arguments except the script name (sys.argv[0])
        predict(sys.argv[1:])