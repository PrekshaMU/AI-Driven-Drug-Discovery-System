import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("drug_dataset.csv")

# Convert text columns into numbers
encoder = LabelEncoder()
for column in data.columns:
    data[column] = encoder.fit_transform(data[column].astype(str))

# Split features and label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "drug_model.pkl")

print("Model trained successfully!")
print("Model saved as drug_model.pkl")
