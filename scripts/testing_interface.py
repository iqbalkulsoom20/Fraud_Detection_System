import joblib
import pandas as pd

# Load the trained model
model_path = './models/random_forest_model.pkl'
model = joblib.load(model_path)

# Prompt user for input values for all 30 features
print("Welcome to the Credit Card Fraud Detection System!")
input_data = []
for i in range(1, 31):  # Adjust range to match number of features
    value = float(input(f"Enter value for Feature_{i}: "))
    input_data.append(value)

# Convert input data to a DataFrame with appropriate column names
columns = [f"Feature_{i}" for i in range(1, 31)]  # Update column names to match dataset
input_df = pd.DataFrame([input_data], columns=columns)

# Make prediction
prediction = model.predict(input_df)

# Display result
if prediction[0] == 1:
    print("This transaction is fraudulent.")
else:
    print("This transaction is legitimate.")
