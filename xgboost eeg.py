# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
eeg_df = pd.read_csv("C:\Sumukh\prashant sir\EEG.machinelearing_data_BRMH.csv")

# Display basic statistics of the dataset
print(eeg_df.describe())

# Drop duplicate rows
eeg_df = eeg_df.drop_duplicates()

# Display shape of the dataset
print("Dataset shape after removing duplicates:", eeg_df.shape)

# Display columns with object (categorical) data types
print("Categorical columns:", eeg_df.dtypes[eeg_df.dtypes == 'object'])

# Define target variables and feature set
Y1 = eeg_df['main.disorder']
Y2 = eeg_df['specific.disorder']
X = eeg_df.drop(['main.disorder', 'specific.disorder', 'eeg.date'], axis=1)

# Display shape of feature set
print("Feature set shape:", X.shape)

# Display categorical columns in the feature set
print("Categorical columns in features:", X.dtypes[X.dtypes == 'object'])

# Drop columns with all missing values
X = X.dropna(axis=1, how='all')

# Convert categorical variable 'sex' to dummy/one-hot encoded variables
X1 = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X1_imputed = imputer.fit_transform(X1)

# Standardize the features
scaler = StandardScaler()
X1_rescaled = scaler.fit_transform(X1_imputed)

# Perform PCA to retain components explaining 95% of variance
pca = PCA(n_components=0.95)
pca.fit(X1_rescaled)
reduced = pca.transform(X1_rescaled)
print("Number of components explaining 95% variance:", pca.n_components_)
print("Shape of reduced feature set:", reduced.shape)

# Encode the target variable Y1 into numeric labels
label_encoder = LabelEncoder()
Y1_encoded = label_encoder.fit_transform(Y1)

# Display mapping of labels for reference
print("Class Mapping:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index}: {class_name}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reduced, Y1_encoded, test_size=0.2, random_state=42, stratify=Y1_encoded
)

# Initialize and train the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
