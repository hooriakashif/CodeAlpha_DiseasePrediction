import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "heart.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded for model training")
print("Shape:", df.shape)

# Split features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    results[name] = acc
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

# Show summary
print("\n✅ Training complete! Accuracy summary:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# ✅ Save the best model (Random Forest)
best_model_name = max(results, key=results.get)
best_model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name.replace(' ', '_')}.pkl"))
joblib.dump(best_model, best_model_path)

print(f"\n✅ Best model ({best_model_name}) saved successfully as {best_model_path}")
