import pandas as pd
import joblib
import os

# Load the saved model and scaler
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
model = joblib.load(os.path.join(MODELS_DIR, "trained_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Function to make prediction
def predict_disease(features):
    # Convert input features to DataFrame
    input_df = pd.DataFrame([features], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100  # Probability of having disease
    
    return prediction, probability

# Example usage (you can replace with real user input)
if __name__ == "__main__":
    print("✅ Model and scaler loaded successfully!")
    
    # Sample input: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    sample_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]  # Another sample (should predict 1)
    
    pred, prob = predict_disease(sample_features)
    result = "has heart disease" if pred == 1 else "does not have heart disease"
    print(f"\nPrediction: The patient {result} (Probability: {prob:.2f}%)")