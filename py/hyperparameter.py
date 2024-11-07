import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import preprocessing
import modeling

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv("preprocessed_breast_cancer_data.csv")

    # Separate features and target
    X = df.drop('Status', axis=1).values
    y = df['Status'].values

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision Tree parameter grid
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Neural Network parameter grid
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [300]
    }

    # 1. Decision Tree Grid Search
    print("\nPerforming Grid Search for Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(
        dt,
        dt_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    dt_grid.fit(X_train, y_train)

    # 2. Neural Network Grid Search
    print("\nPerforming Grid Search for Neural Network...")
    nn = MLPClassifier(random_state=42, activation='relu')
    nn_grid = GridSearchCV(
        nn,
        nn_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    nn_grid.fit(X_train, y_train)

    # Print results for both models
    models = {
        'Decision Tree': dt_grid,
        'Neural Network': nn_grid
    }

    for name, model in models.items():
        print(f"\n{name} Results:")
        print("Best parameters:", model.best_params_)
        print("Best cross-validation score:", model.best_score_)

        y_pred = model.predict(X_test)
        print("\nTest Set Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame({
        'Model': ['Decision Tree', 'Neural Network'],
        'Best Parameters': [str(dt_grid.best_params_), str(nn_grid.best_params_)],
        'Best CV Score': [dt_grid.best_score_, nn_grid.best_score_],
        'Test Accuracy': [
            accuracy_score(y_test, dt_grid.predict(X_test)),
            accuracy_score(y_test, nn_grid.predict(X_test))
        ]
    })

    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("\nResults saved to 'hyperparameter_tuning_results.csv'")

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.close()

def main():
    print("Loading preprocessed data...")
    df = pd.read_csv("preprocessed_breast_cancer_data.csv")

    X = df.drop('Status', axis=1).values
    y = df['Status'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Decision Tree parameter using grid
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #Neural Network parameter using grid
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [300]
    }

    #Decision Tree
    print("\nPerforming Grid Search for Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(
        dt,
        dt_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    dt_grid.fit(X_train, y_train)

    #Neural Network
    print("\nPerforming Grid Search for Neural Network...")
    nn = MLPClassifier(random_state=42, activation='relu')
    nn_grid = GridSearchCV(
        nn,
        nn_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    nn_grid.fit(X_train, y_train)

    #both models
    models = {
        'Decision Tree': dt_grid,
        'Neural Network': nn_grid
    }

    for name, model in models.items():
        print(f"\n{name} Results:")
        print("Best parameters:", model.best_params_)
        print("Best cross-validation score:", model.best_score_)

        y_pred = model.predict(X_test)

        print("\nTest Set Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))

        #confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame({
        'Model': ['Decision Tree', 'Neural Network'],
        'Best Parameters': [str(dt_grid.best_params_), str(nn_grid.best_params_)],
        'Best CV Score': [dt_grid.best_score_, nn_grid.best_score_],
        'Test Accuracy': [
            accuracy_score(y_test, dt_grid.predict(X_test)),
            accuracy_score(y_test, nn_grid.predict(X_test))
        ]
    })

    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("\nResults saved to 'hyperparameter_tuning_results.csv'")

if __name__ == "__main__":
    main()
