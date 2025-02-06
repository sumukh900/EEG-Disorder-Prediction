import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
eeg_df = pd.read_csv("C:\Sumukh\prashant sir\EEG.machinelearing_data_BRMH.csv")

# Initial data exploration
print(eeg_df.describe())
print(eeg_df.dtypes)

# Drop duplicates
eeg_df = eeg_df.drop_duplicates()

# Check data shape
print("Data shape after removing duplicates:", eeg_df.shape)

# Separate target and features
Y1 = eeg_df['main.disorder']
Y2 = eeg_df['specific.disorder']
X = eeg_df.drop(['main.disorder', 'specific.disorder', 'eeg.date'], axis=1)

# Handle categorical variables
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Replace NaN values with mean (mean imputation)
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_rescaled = scaler.fit_transform(X)

# Reduce dimensionality with PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_rescaled)
print("Number of components explaining 95% variance:", pca.n_components_)
print("Shape of reduced feature set:", X_reduced.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, Y1, test_size=0.2, random_state=42, stratify=Y1
)

# Initialize the CatBoost classifier
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    random_seed=42
)

# Train the model
catboost_model.fit(X_train, y_train)

# Evaluate the model
y_pred = catboost_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Feature importance (if applicable)
feature_importances = catboost_model.get_feature_importance()
print("\nFeature Importances:", feature_importances)
