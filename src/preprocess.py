import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Loads the Iris dataset, splits it into training and testing sets,
    and returns both scaled and raw features along with targets.
    """
    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 2. Split dataset (80% train, 20% test)
    # stratify=y ensures equal class distribution across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Feature Scaling (Crucial for distance-based models like KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Return everything needed for training and evaluation
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "target_names": iris.target_names
    }

if __name__ == "__main__":
    # Test if the script runs fine on its own
    data = load_and_preprocess_data()
    print("Data preparation successful!")
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Testing set shape: {data['X_test'].shape}")