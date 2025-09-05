import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Define column names for the UCI Heart Disease dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load the dataset
try:
    # Assuming the .data file is comma-separated and has no header
    df = pd.read_csv('heart.data', sep=',', header=None, names=column_names, na_values='?')
    print("Dataset heart.data loaded successfully.")
except FileNotFoundError:
    print("Error: heart.data not found. Please make sure the file is in the same directory as train.py")
    exit()

print("\n--- DATA DIAGNOSTICS ---")

# Check original class distribution
print("Original target distribution:")
print(df['target'].value_counts().sort_index())
print("\nOriginal target percentages:")
print((df['target'].value_counts(normalize=True).sort_index() * 100).round(2))

# Convert to binary classification (0 = No Disease, 1 = Disease Present)
df['target_binary'] = (df['target'] > 0).astype(int)

print("\nBinary target distribution:")
print(df['target_binary'].value_counts().sort_index())
print("\nBinary target percentages:")
print((df['target_binary'].value_counts(normalize=True).sort_index() * 100).round(2))

# --- EDA ---
print("\n--- Exploratory Data Analysis ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Check for data quality issues
print("\n--- DATA QUALITY CHECKS ---")
print("Feature value ranges:")
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    print(f"{col}: {df[col].min():.1f} to {df[col].max():.1f}")

print("\nCategorical feature unique values:")
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    print(f"{col}: {sorted(df[col].unique())}")

# --- Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Define categorical and numerical features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Check if all defined features exist in the DataFrame
all_features = categorical_features + numerical_features + ['target', 'target_binary']
if not all(feature in df.columns for feature in all_features):
    missing = [f for f in all_features if f not in df.columns]
    print(f"Error: Some expected columns are missing from the dataset: {missing}")
    exit()

# Handle missing values (if any)
if df.isnull().sum().sum() > 0:
    print("Warning: Missing values detected. Imputing with median for numerical features.")
    for col in numerical_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
else:
    print("No missing values found. Skipping missing value imputation.")

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Separate features and target variable (use binary target)
X = df.drop(['target', 'target_binary'], axis=1)
y = df['target_binary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data split into training (80%) and testing (20%) sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Training set class distribution:")
print(y_train.value_counts().sort_index())
print(f"Test set class distribution:")
print(y_test.value_counts().sort_index())

# --- Improved Model Training ---
print("\n--- IMPROVED MODEL TRAINING ---")

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Computed class weights: {class_weight_dict}")

# Try multiple models
models_to_try = {
    'Logistic Regression (Balanced)': LogisticRegression(
        solver='liblinear', 
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    ),
    'Logistic Regression (Custom Weights)': LogisticRegression(
        solver='liblinear', 
        random_state=42,
        class_weight=class_weight_dict,
        max_iter=1000
    ),
    'Random Forest (Balanced)': RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100
    ),
    'Calibrated Random Forest': CalibratedClassifierCV(
        RandomForestClassifier(random_state=42, n_estimators=100),
        method='isotonic',
        cv=5
    )
}

best_model = None
best_score = 0
best_name = ""

for name, model in models_to_try.items():
    print(f"\n--- Training {name} ---")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Check prediction distribution
    print(f"Predictions: {np.bincount(y_pred)}")
    print(f"Probability ranges: [{y_pred_proba[:, 1].min():.3f}, {y_pred_proba[:, 1].max():.3f}]")
    
    # Use F1 score for model selection (better for imbalanced data)
    if f1 > best_score:
        best_score = f1
        best_model = pipeline
        best_name = name

print(f"\n--- Best Model: {best_name} (F1: {best_score:.4f}) ---")

# Use the best model as your final model
model_pipeline = best_model

# --- FINAL MODEL EVALUATION ---
print("\n--- FINAL MODEL EVALUATION ---")
y_pred_final = model_pipeline.predict(X_test)
y_pred_proba_final = model_pipeline.predict_proba(X_test)

print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Final Model Precision: {precision_score(y_test, y_pred_final, average='weighted'):.4f}")
print(f"Final Model Recall: {recall_score(y_test, y_pred_final, average='weighted'):.4f}")
print(f"Final Model F1-score: {f1_score(y_test, y_pred_final, average='weighted'):.4f}")
print(f"Final Model ROC-AUC: {roc_auc_score(y_test, y_pred_proba_final[:, 1]):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Check probability distribution by true class
print("\nProbability distributions by true class:")
for class_val in [0, 1]:
    mask = y_test == class_val
    probs = y_pred_proba_final[mask, 1]  # Disease probability
    print(f"Class {class_val}: mean={probs.mean():.3f}, std={probs.std():.3f}, range=[{probs.min():.3f}, {probs.max():.3f}]")

# --- Model Diagnostics ---
print("\n--- MODEL DIAGNOSTICS ---")

# Check model performance on training data
y_train_pred = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)

print("Training set predictions distribution:")
print(pd.Series(y_train_pred).value_counts().sort_index())

print("\nTraining set probability ranges:")
print(f"Class 0 probabilities: {y_train_proba[:, 0].min():.3f} to {y_train_proba[:, 0].max():.3f}")
print(f"Class 1 probabilities: {y_train_proba[:, 1].min():.3f} to {y_train_proba[:, 1].max():.3f}")

print(f"\nModel classes: {model_pipeline.classes_}")

# Check for class imbalance handling
try:
    classifier = model_pipeline.named_steps['classifier']
    if hasattr(classifier, 'class_weight'):
        print(f"Class weights: {classifier.class_weight}")
    else:
        print("No class weighting applied")
except:
    print("Could not check class weights")

# --- Model Saving ---
print("\n--- Model Saving ---")
joblib.dump(model_pipeline, 'heart_disease_model.pkl')
print(f"Best model ({best_name}) saved to heart_disease_model.pkl")

print("\ntrain.py execution complete.")