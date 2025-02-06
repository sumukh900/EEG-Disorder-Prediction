# EEG-Disorder-Prediction

## Project Overview
A comprehensive machine learning project that analyzes EEG (Electroencephalogram) data to classify various mental health disorders. The project implements multiple classification models to identify both main disorders and specific conditions, utilizing advanced preprocessing techniques and machine learning algorithms.

## Features
- Multi-class classification for 7 main disorders and 12 specific disorders
- Implementation of multiple machine learning models:
  - Random Forest
  - XGBoost
  - CatBoost
  - Artificial Neural Networks
- Advanced data preprocessing pipeline including:
  - PCA dimensionality reduction
  - SMOTE for handling class imbalance
  - Feature selection and importance analysis
  - Data standardization and imputation

## Disorder Categories

### Main Disorders
- Addictive disorder
- Anxiety disorder
- Healthy control
- Mood disorder
- Obsessive compulsive disorder
- Schizophrenia
- Trauma and stress related disorder

### Specific Disorders
- Acute stress disorder
- Adjustment disorder
- Alcohol use disorder
- Behavioral addiction disorder
- Bipolar disorder
- Depressive disorder
- Healthy control
- Obsessive compulsive disorder
- Panic disorder
- Posttraumatic stress disorder
- Schizophrenia
- Social anxiety disorder

## Technical Stack
- Python 3.x
- Libraries:
  - scikit-learn
  - TensorFlow/Keras
  - XGBoost
  - CatBoost
  - pandas
  - numpy
  - imbalanced-learn
  - matplotlib/seaborn

## Project Structure
```
├── eeg_up.py              # Main implementation with Random Forest
├── ml_methods.py          # Various ML model implementations
├── xgboost_eeg.py        # XGBoost implementation
├── ANN_eeg.py            # Neural Network implementation
├── catboost_eeg.py       # CatBoost implementation
└── eeg_analysis.py       # Comprehensive analysis script
```

## Model Performance
Each model has been evaluated using standard metrics including:
- Classification accuracy
- Precision, recall, and F1-score
- Balanced accuracy for handling class imbalance

## Data Requirements
The project expects an EEG dataset with the following:
- EEG measurements
- Patient demographic information (including sex)
- Main disorder labels
- Specific disorder labels

Note: The actual EEG dataset is not included in this repository due to privacy concerns.

## Results
The project achieves classification of mental health disorders using EEG data through:
- Feature selection to identify most relevant EEG patterns
- Handling of class imbalance using SMOTE
- Comparison of multiple machine learning approaches
- Optimization of model parameters using GridSearchCV

