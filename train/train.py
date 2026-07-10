import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Clean data
df.columns = df.columns.str.strip()
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# Drop unwanted column
df.drop("loan_id", axis=1, inplace=True)

# Encode categorical columns
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_status = LabelEncoder()

df["education"] = le_education.fit_transform(df["education"])
df["self_employed"] = le_self_employed.fit_transform(df["self_employed"])
df["loan_status"] = le_status.fit_transform(df["loan_status"])

# Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Encoders
with open("education_encoder.pkl", "wb") as file:
    pickle.dump(le_education, file)

with open("self_employed_encoder.pkl", "wb") as file:
    pickle.dump(le_self_employed, file)

with open("status_encoder.pkl", "wb") as file:
    pickle.dump(le_status, file)

print("Model and encoders saved successfully!")