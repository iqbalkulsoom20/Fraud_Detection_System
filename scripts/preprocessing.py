import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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
# Example: Remove rows with missing values in the target column
df = df.dropna(subset=['Class'])

# Apply SMOTE to handle class imbalance (oversample the minority class)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Print the shape of the training and testing sets to confirm
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Save the resampled dataset (optional)
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['Class'] = y_resampled
resampled_df.to_csv('./datasets/creditcard_fraud_detection_resampled.csv', index=False)

# Return the preprocessed data
print("Preprocessing complete.")
