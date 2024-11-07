import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import preprocessing
import modeling
import hyperparameter

def get_feature_importance(models, X, y, feature_names):
    """Calculate feature importance for each model."""
    importance_df = pd.DataFrame()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"\nCalculating feature importance for {name}...")

        model.fit(X_train, y_train)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importances = result.importances_mean

        importance_df[name] = importances

    importance_df.index = feature_names
    importance_df['Mean Importance'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('Mean Importance', ascending=False)

    return importance_df

def plot_feature_importance(importance_df, top_n=10):
    """Plot top N most important features."""
    plt.figure(figsize=(12, 6))

    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['Mean Importance'])
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Mean Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()

def main():
    print("Loading preprocessed data...")
    df = pd.read_csv("preprocessed_breast_cancer_data.csv")

    X = df.drop('Status', axis=1)
    y = df['Status']

    #Initialize all models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42)
    }

    #feature importance
    importance_df = get_feature_importance(models, X.values, y, X.columns)

    print("\nFeature Importance Rankings:")
    print(importance_df)

    importance_df.to_csv('feature_importance.csv')
    print("\nFeature importance rankings saved to 'feature_importance.csv'")

    plot_feature_importance(importance_df)

if __name__ == "__main__":
    main()
