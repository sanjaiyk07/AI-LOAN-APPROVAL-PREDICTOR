import pickle
import pandas as pd

# Load Model and Encoders
model = pickle.load(open("loan_model.pkl", "rb"))
le_education = pickle.load(open("education_encoder.pkl", "rb"))
le_self_employed = pickle.load(open("self_employed_encoder.pkl", "rb"))
le_status = pickle.load(open("status_encoder.pkl", "rb"))

print("\n===== LOAN APPROVAL PREDICTION =====\n")

# User Input
no_of_dependents = int(input("Enter Number of Dependents: "))

education_input = input(
    "Enter Education (Graduate/Not Graduate): "
).strip()
education = le_education.transform([education_input])[0]

self_emp_input = input(
    "Self Employed (Yes/No): "
).strip()
self_employed = le_self_employed.transform([self_emp_input])[0]

income_annum = int(input("Enter Annual Income: "))
loan_amount = int(input("Enter Loan Amount: "))
loan_term = int(input("Enter Loan Term (Years): "))
cibil_score = int(input("Enter CIBIL Score: "))

residential_assets_value = int(
    input("Enter Residential Assets Value: ")
)

commercial_assets_value = int(
    input("Enter Commercial Assets Value: ")
)

luxury_assets_value = int(
    input("Enter Luxury Assets Value: ")
)

bank_asset_value = int(
    input("Enter Bank Asset Value: ")
)

# Create Input DataFrame
input_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

# Predict
prediction = model.predict(input_data)

# Convert Prediction Back to Original Label
result = le_status.inverse_transform(prediction)

print("\n===================================")
print("Loan Status:", result[0])

if result[0] == "Approved":
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")
print("===================================")