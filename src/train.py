import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import the data preparation function from our preprocessing module
from preprocess import load_and_preprocess_data

def train_and_evaluate():
    # 1. Fetch the preprocessed data
    print("Loading and preparing data...")
    data = load_and_preprocess_data()
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    target_names = data["target_names"]

    # 2. Initialize the models
    models = {
        "K-Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    # 3. Train and evaluate each model loop
    for name, model in models.items():
        print("\n" + "="*50)
        print(f"Training Model: {name.replace('_', ' ')}")
        print("="*50)
        
        # KNN needs scaled data; Trees use raw data (though scaled won't hurt them)
        if "Neighbors" in name:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=target_names))
        
        # 4. Generate and save Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {name.replace("_", " ")}')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        
        # Save the figure to the outputs directory
        # Save the figure to the outputs directory
        output_path = os.path.join("..", "outputs", f"{name}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved confusion matrix plot to: {output_path}")

    # 5. Final Summary Comparison
    print("\n" + "="*50)
    print("Final Model Comparison Summary")
    print("="*50)
    for name, acc in results.items():
        print(f"{name.replace('_', ' ')}: {acc * 100:.2f}% Accuracy")

if __name__ == "__main__":
    train_and_evaluate()