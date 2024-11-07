import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tensorflow as tf
import preprocessing

class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances between x and all examples in the training set
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Get most common label
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, f"{model_name} Confusion Matrix")

    return accuracy, cm, report

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv("preprocessed_breast_cancer_data.csv")

    # Separate features and target
    X = df.drop('Status', axis=1).values
    y = df['Status'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Dictionary to store results
    results = {}

    # 1. KNN from scratch
    print("\nTraining KNN from scratch...")
    knn = KNNFromScratch(k=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = evaluate_model(y_test, y_pred_knn, "KNN")

    # 2. Naive Bayes
    print("\nTraining Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['Naive Bayes'] = evaluate_model(y_test, y_pred_nb, "Naive Bayes")

    # 3. C4.5 Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results['Decision Tree'] = evaluate_model(y_test, y_pred_dt, "Decision Tree")

    # 4. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # 5. Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results['Gradient Boosting'] = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

    # 6. Neural Network
    print("\nTraining Neural Network...")
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    results['Neural Network'] = evaluate_model(y_test, y_pred_nn, "Neural Network")

    # Compare all models
    print("\nModel Comparison:")
    accuracies = {model: result[0] for model, result in results.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Model Accuracies Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # Save results to CSV
    comparison_df = pd.DataFrame({
        'Model': accuracies.keys(),
        'Accuracy': accuracies.values()
    })
    comparison_df.to_csv('model_comparison.csv', index=False)

    print("\nResults saved to 'model_comparison.csv'")
    print("Confusion matrices and model comparison plots have been saved as PNG files")

if __name__ == "__main__":
    main()
