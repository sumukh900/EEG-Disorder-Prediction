import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline  # Added this import
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("C:\Sumukh\prashant sir\EEG.machinelearing_data_BRMH.csv")

# Initial data exploration
print("\nData Description:")
print(data.describe())

# Remove duplicates
data = data.drop_duplicates()
print("\nShape after removing duplicates:", data.shape)

# Check object datatypes
print("\nColumns with object dtype:")
print(data.dtypes[data.dtypes == 'object'])

# Convert sex to numeric using LabelEncoder
print("\nConverting sex to numeric...")
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])

# Remove 'eeg.date' column and target variables, keep 'sex'
X = data.drop(['eeg.date', 'main.disorder', 'specific.disorder'], axis=1)
print("\nFeature matrix shape:", X.shape)

# Check for any remaining object columns
print("\nRemaining object columns:")
print(X.dtypes[X.dtypes == 'object'])

# Convert sex to dummy variables
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Convert all columns to numeric, replacing errors with NaN
numeric_columns = X.columns
X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Get target variables
y_main = data['main.disorder']
y_specific = data['specific.disorder']

# Create a pipeline for preprocessing
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # First handle missing values
    ('scaler', StandardScaler()),                 # Then scale the features
    ('pca', PCA(n_components=0.95))              # Finally apply PCA
])

# Fit and transform the data using the preprocessing pipeline
print("\nApplying preprocessing pipeline...")
X_processed = preprocessing_pipeline.fit_transform(X)

print("\nPCA Results:")
print("Number of components:", preprocessing_pipeline.named_steps['pca'].n_components_)
print("Processed data shape:", X_processed.shape)

# Define the main disorder categories
main_disorder_categories = [
    'Addictive disorder',
    'Anxiety disorder',
    'Healthy control',
    'Mood disorder',
    'Obsessive compulsive disorder',
    'Schizophrenia',
    'Trauma and stress related disorder'
]

# Define the specific disorder categories
specific_disorder_categories = [
    'Acute stress disorder',
    'Adjustment disorder',
    'Alcohol use disorder',
    'Behavioral addiction disorder',
    'Bipolar disorder',
    'Depressive disorder',
    'Healthy control',
    'Obsessive compulsitve disorder',
    'Panic disorder',
    'Posttraumatic stress disorder',
    'Schizophrenia',
    'Social anxiety disorder'
]

# Create optimized pipeline for classification
pipeline_main = ImbPipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        warm_start=True
    ))
])

# Simplified parameter grid
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5]
}

print("\nProcessing main disorders...")

# Split data using the processed features
X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
    X_processed, y_main, test_size=0.2, random_state=42, stratify=y_main
)

print("\nDataset splits:")
print(f"Training set size: {len(X_train_main)}")
print(f"Test set size: {len(X_test_main)}")
print(f"Training labels: {len(y_train_main)}")
print(f"Test labels: {len(y_test_main)}")

# Perform grid search with reduced cross-validation
grid_search = GridSearchCV(
    pipeline_main,
    param_grid,
    cv=3,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model for main disorders
print("\nTraining main disorder classifier...")
grid_search.fit(X_train_main, y_train_main)

# Make predictions for main disorders
y_pred_main = grid_search.predict(X_test_main)

# Print main disorder classification report
print("\nMain Disorder Classification Report")
print(classification_report(
    y_test_main, 
    y_pred_main,
    labels=main_disorder_categories,
    zero_division=0,
    digits=2
))

print("\nProcessing specific disorders...")

# Create pipeline for specific disorders
pipeline_specific = ImbPipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        warm_start=True
    ))
])

# Split data for specific disorders
X_train_specific, X_test_specific, y_train_specific, y_test_specific = train_test_split(
    X_processed, y_specific, test_size=0.2, random_state=42, stratify=y_specific
)

# Grid search for specific disorders
grid_search_specific = GridSearchCV(
    pipeline_specific,
    param_grid,
    cv=3,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model for specific disorders
print("Training specific disorder classifier...")
grid_search_specific.fit(X_train_specific, y_train_specific)

# Make predictions for specific disorders
y_pred_specific = grid_search_specific.predict(X_test_specific)

# Print specific disorder classification report
print("\nSpecific Disorder Classification Report")
print(classification_report(
    y_test_specific, 
    y_pred_specific,
    labels=specific_disorder_categories,
    zero_division=0,
    digits=2
))

print("\nOptimal parameters for main disorders:", grid_search.best_params_)
print("Optimal parameters for specific disorders:", grid_search_specific.best_params_)

# Print sex distribution in the dataset
print("\nSex distribution in the dataset:")
print(data['sex'].value_counts(normalize=True) * 100)
