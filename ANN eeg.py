# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
eeg_df = pd.read_csv("C:\Sumukh\prashant sir\EEG.machinelearing_data_BRMH.csv")

# Display basic statistics of the dataset
print(eeg_df.describe())

# Drop duplicate rows
eeg_df = eeg_df.drop_duplicates()

# Display shape of the dataset
print("Dataset shape after removing duplicates:", eeg_df.shape)

# Define target variables and feature set
Y1 = eeg_df['main.disorder']
Y2 = eeg_df['specific.disorder']
X = eeg_df.drop(['main.disorder', 'specific.disorder', 'eeg.date'], axis=1)

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
pca = PCA(n_components=0.99)
pca.fit(X1_rescaled)
reduced = pca.transform(X1_rescaled)
print("Number of components explaining 95% variance:", pca.n_components_)
print("Shape of reduced feature set:", reduced.shape)

# Encode the target variable Y1 into numeric labels
label_encoder = LabelEncoder()
Y1_encoded = label_encoder.fit_transform(Y1)

# Convert labels to categorical for ANN
Y1_categorical = to_categorical(Y1_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reduced, Y1_categorical, test_size=0.2, random_state=42, stratify=Y1_encoded
)

# Build the ANN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(Y1_categorical.shape[1], activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on the test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluate the predictions
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("Accuracy Score:", accuracy_score(y_true, y_pred))
