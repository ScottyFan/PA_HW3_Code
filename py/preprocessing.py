import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df

def encode_categorical_variables(df):
    label_encoder = LabelEncoder()
    categorical_columns = [
        'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage',
        'differentiate', 'Grade', 'A Stage', 'Estrogen Status',
        'Progesterone Status', 'Status'
    ]

    encoded_df = df.copy()
    for column in categorical_columns:
        if column in encoded_df.columns:
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column].astype(str))

    return encoded_df

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

def standardize_features(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
    return scaled_df

def apply_pca(df, n_components=None, variance_threshold=0.95):
    if n_components is None:
        pca = PCA()
        pca.fit(df)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)

    transformed_data = pca.fit_transform(df)
    columns = [f'PC{i+1}' for i in range(n_components)]
    transformed_df = pd.DataFrame(transformed_data, columns=columns)

    return transformed_df, pca, pca.explained_variance_ratio_

def plot_explained_variance(explained_variance_ratio):
    plt.figure(figsize=(10, 6))
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()

def main():
    print("Loading data...")
    input_file = "Breast_Cancer_dataset.csv"
    df = load_data(input_file)
    original_shape = df.shape

    target = df['Status'].copy()
    df = df.drop('Status', axis=1)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Encoding categorical variables...")
    df = encode_categorical_variables(df)

    print("Removing outliers...")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df = remove_outliers(df, numeric_columns)

    print("Standardizing features...")
    df_scaled = standardize_features(df)

    print("Applying PCA...")
    transformed_df, pca, explained_variance_ratio = apply_pca(df_scaled)

    plot_explained_variance(explained_variance_ratio)

    final_df = transformed_df.copy()
    final_df['Status'] = target[transformed_df.index]

    output_file = "preprocessed_breast_cancer_data.csv"
    final_df.to_csv(output_file, index=False)

    print("\nPreprocessing Summary:")
    print(f"Original dataset shape: {original_shape}")
    print(f"Preprocessed dataset shape: {final_df.shape}")
    print(f"\nNumber of PCA components: {transformed_df.shape[1]}")
    print("\nExplained variance ratio by component:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")
    print(f"\nTotal variance explained: {sum(explained_variance_ratio):.4f}")

    print("\nMissing values in preprocessed data:")
    print(final_df.isnull().sum())

    print("\nSaved preprocessed data to:", output_file)
    print("Saved PCA explained variance plot to: pca_explained_variance.png")

if __name__ == "__main__":
    main()
