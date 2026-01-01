from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import json
import numpy as np

def main():
    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Ensures balanced splits
    )

    # Train model
    model = LogisticRegression(max_iter=200, random_state=42)  # Added random_state for reproducibility
    model.fit(X_train, y_train)

    # Evaluate model
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    # Save model and artifacts
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Enhanced metrics
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "model_params": model.get_params()
    }
    
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Detailed reporting
    print(f"Model saved to {model_path}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
