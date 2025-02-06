import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("C:\Sumukh\prashant sir\EEG.machinelearing_data_BRMH.csv")

# Convert sex to numeric using LabelEncoder
print("Converting sex to numeric...")
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])

# Remove 'eeg.date' column and target variables, keep 'sex'
X = data.drop(['eeg.date', 'main.disorder', 'specific.disorder'], axis=1)

# Convert all columns to numeric, replacing errors with NaN
numeric_columns = X.columns.drop('sex')  # Exclude 'sex' from conversion
X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Get target variables
y_main = data['main.disorder']
y_specific = data['specific.disorder']

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

# Create optimized pipeline with reduced complexity
pipeline_main = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=11)),  # Increased k to include sex
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

print("Processing main disorders...")

# Split data
X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
    X, y_main, test_size=0.2, random_state=42, stratify=y_main
)

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
print("Training main disorder classifier...")
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
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=11)),  # Keep original k value
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
    X, y_specific, test_size=0.2, random_state=42, stratify=y_specific
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

# Print selected features
feature_selector = grid_search.best_estimator_.named_steps['feature_selection']
selected_mask = feature_selector.get_support()
selected_features = X.columns[selected_mask].tolist()
print("\nSelected features (including sex if selected):")
for feature in selected_features:
    print(feature)

# Print sex-specific feature importance information
if 'sex' in selected_features:
    print("\nSex was selected as an important feature")
else:
    print("\nSex was not selected as one of the top important features")

print("\nOptimal parameters for main disorders:", grid_search.best_params_)
print("Optimal parameters for specific disorders:", grid_search_specific.best_params_)

# Print sex distribution in the dataset
print("\nSex distribution in the dataset:")
print(data['sex'].value_counts(normalize=True) * 100)
