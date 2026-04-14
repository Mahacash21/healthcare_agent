import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ══════════════════════════════════════════════
# STEP 1 — Load Data
# ══════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv("claims.csv")
print(f"Loaded {len(df):,} claims")

# ══════════════════════════════════════════════
# STEP 2 — Prepare Features
# ══════════════════════════════════════════════
print("\nPreparing features...")

# These are our input columns (what we know about a claim)
feature_columns = [
    'procedure_code',
    'diagnosis_code',
    'provider_type',
    'payer_type',
    'place_of_service',
    'patient_age',
    'claim_amount',
    'num_diagnoses',
    'prior_auth',
    'is_specialist'
]

# This is what we want to predict
target_column = 'denied'

# Separate features (X) from target (y)
# X = everything the model learns FROM
# y = what the model tries to PREDICT
X = df[feature_columns]
y = df[target_column]

print(f"Features: {len(feature_columns)}")
print(f"Target: {target_column}")

# ══════════════════════════════════════════════
# STEP 3 — Encode Text Columns
# ══════════════════════════════════════════════
# Machine learning models only understand numbers
# We need to convert text like 'Medicaid' into numbers
print("\nEncoding text columns...")

text_columns = [
    'procedure_code',
    'diagnosis_code',
    'provider_type',
    'payer_type',
    'place_of_service'
]

# LabelEncoder converts each unique text value to a number
# e.g. 'Medicaid'=0, 'Commercial'=1, 'Medicare FFS'=2
encoders = {}
X = X.copy()

for col in text_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

# ══════════════════════════════════════════════
# STEP 4 — Split into Train and Test Sets
# ══════════════════════════════════════════════
# We train the model on 80% of data
# We test it on the remaining 20% it has never seen
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # makes split reproducible
)

print(f"Training set:  {len(X_train):,} claims")
print(f"Testing set:   {len(X_test):,} claims")

# ══════════════════════════════════════════════
# STEP 5 — Train the Model
# ══════════════════════════════════════════════
print("\nTraining Random Forest model...")
print("(This may take a few seconds...)")

model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,    # reproducible results
    n_jobs=-1           # use all CPU cores
)

model.fit(X_train, y_train)
print("Model trained!")

# ══════════════════════════════════════════════
# STEP 6 — Evaluate the Model
# ══════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Make predictions on test set
y_pred = model.predict(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.1%}")

# Detailed report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Approved', 'Denied']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted Approved  Predicted Denied")
print(f"Actual Approved       {cm[0][0]:,}              {cm[0][1]:,}")
print(f"Actual Denied         {cm[1][0]:,}              {cm[1][1]:,}")

# ══════════════════════════════════════════════
# STEP 7 — Feature Importance
# ══════════════════════════════════════════════
print("\n--- What Matters Most for Denial Prediction ---")

importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:<20} {row['importance']:.1%}  {bar}")

# ══════════════════════════════════════════════
# STEP 8 — Test a Single Prediction
# ══════════════════════════════════════════════
print("\n--- Test Prediction on a Single Claim ---")

# Create a sample claim
sample_claim = pd.DataFrame([{
    'procedure_code':   '27447',   # knee replacement
    'diagnosis_code':   'M79.3',   # pain in forearm
    'provider_type':    'Orthopedics',
    'payer_type':       'Medicare Advantage',
    'place_of_service': 'Inpatient Hospital',
    'patient_age':      68,
    'claim_amount':     8500.00,
    'num_diagnoses':    3,
    'prior_auth':       0,          # no prior auth!
    'is_specialist':    1
}])

# Encode text columns using same encoders
for col in text_columns:
    sample_claim[col] = encoders[col].transform(sample_claim[col])

# Predict
prediction = model.predict(sample_claim)[0]
probability = model.predict_proba(sample_claim)[0]

print(f"\nClaim Details:")
print(f"  Procedure:    Knee Replacement (27447)")
print(f"  Payer:        Medicare Advantage")
print(f"  Amount:       $8,500")
print(f"  Prior Auth:   No")
print(f"  Provider:     Orthopedics")

print(f"\nPrediction:   {'DENIED' if prediction == 1 else 'APPROVED'}")
print(f"Confidence:   {max(probability):.1%}")
print(f"Denial risk:  {probability[1]:.1%}")

# ══════════════════════════════════════════════
# STEP 9 — Save the Model
# ══════════════════════════════════════════════
import pickle

print("\nSaving model...")
with open('claims_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model saved to claims_model.pkl")
print("Encoders saved to encoders.pkl")
print("\nDone! Ready to build the prediction UI.")