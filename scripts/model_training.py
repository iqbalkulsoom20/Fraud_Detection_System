import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load the dataset (from the 'datasets' folder)
file_path = './datasets/creditcard_fraud_detection.csv'
df = pd.read_csv(file_path)

# Check if the dataset loaded correctly
print("Dataset loaded successfully:")
print(df.head())

# Split the dataset into features (X) and target (y)
X = df.drop('Class', axis=1)  # Drop the 'Class' column for features
y = df['Class']  # Target column is 'Class'

# Handle missing values if necessary
df = df.dropna(subset=['Class'])

# Apply SMOTE to handle class imbalance (oversample the minority class)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training set
rf_model.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create the 'models' directory if it doesn't exist
models_directory = './models'
if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Save the trained model
joblib.dump(rf_model, os.path.join(models_directory, 'random_forest_model.pkl'))

print("\nModel training complete.")

