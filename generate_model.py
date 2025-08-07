import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate sample customer data
data = {
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
    'tenure': np.random.randint(1, 73, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
    'MonthlyCharges': np.random.uniform(18, 120, n_samples),
    'TotalCharges': np.random.uniform(18, 9000, n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples)
}

df = pd.DataFrame(data)

# Prepare features and target
X = df.drop('Churn', axis=1)
y = (df['Churn'] == 'Yes').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit encoders for categorical variables
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaperlessBilling', 'PaymentMethod']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(X_train[col])
    encoders[col] = le

# Create and fit scaler for numerical variables
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])

# Transform the data
X_train_encoded = X_train.copy()
for col, encoder in encoders.items():
    X_train_encoded[col] = encoder.transform(X_train_encoded[col])

X_train_scaled = X_train_encoded.copy()
X_train_scaled[numerical_cols] = scaler.transform(X_train_scaled[numerical_cols])

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and preprocessing components
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model files generated successfully!")
print("Files created:")
print("- best_model.pkl")
print("- encoder.pkl") 
print("- scaler.pkl")
