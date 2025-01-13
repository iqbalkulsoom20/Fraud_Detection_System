import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Function to generate synthetic Credit Card Fraud Dataset
def generate_synthetic_dataset():
    # Parameters for the dataset
    n_samples = 10000  # Number of samples (transactions)
    n_features = 30    # Number of features (attributes of each transaction)
    n_classes = 2      # Two classes: 0 (non-fraud), 1 (fraud)
    
    # Create a synthetic dataset using sklearn's make_classification
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_classes=n_classes, 
                               weights=[0.91, 0.09],  # Imbalance between classes
                               flip_y=0,  # No noise in labels
                               random_state=42)
    
    # Convert to DataFrame for easier manipulation and analysis
    feature_columns = [f'Feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['Class'] = y  # Add the target column as 'Class' (fraud or non-fraud)

    # Save to CSV file in the 'datasets' folder
    file_path = './datasets/creditcard_fraud_detection.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Synthetic dataset generated and saved to {file_path}")
    
    return df

# Generate the dataset
generate_synthetic_dataset()
