import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

# ── Healthcare reference data ──────────────────────────
procedure_codes = [
    '99213', '99214', '99232', '93000', '71046',
    '80053', '99283', '27447', '43239', '70553'
]

diagnosis_codes = [
    'I10', 'E11.9', 'J18.9', 'M79.3', 'Z12.11',
    'F32.9', 'K21.0', 'N39.0', 'J06.9', 'R51'
]

provider_types = [
    'Internal Medicine', 'Family Practice', 'Cardiology',
    'Orthopedics', 'Emergency Medicine', 'Radiology',
    'Gastroenterology', 'Psychiatry', 'Urology', 'Neurology'
]

payer_types = [
    'Medicare Advantage', 'Commercial', 'Medicaid', 'Medicare FFS'
]

place_of_service = [
    'Office', 'Outpatient Hospital', 'Emergency Room',
    'Inpatient Hospital', 'Telehealth'
]

# ── Generate base features ─────────────────────────────
df = pd.DataFrame({
    'procedure_code':   np.random.choice(procedure_codes, n),
    'diagnosis_code':   np.random.choice(diagnosis_codes, n),
    'provider_type':    np.random.choice(provider_types, n),
    'payer_type':       np.random.choice(payer_types, n),
    'place_of_service': np.random.choice(place_of_service, n),
    'patient_age':      np.random.randint(18, 90, n),
    'claim_amount':     np.round(np.random.lognormal(5.5, 1.2, n), 2),
    'num_diagnoses':    np.random.randint(1, 8, n),
    'prior_auth':       np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'is_specialist':    np.random.choice([0, 1], n, p=[0.4, 0.6]),
})

# ── Create realistic denial logic ─────────────────────
# (based on real payor denial patterns)
denial_probability = np.zeros(n)

# High cost claims more likely denied
denial_probability += (df['claim_amount'] > 5000).astype(float) * 0.25

# No prior auth increases denial risk
denial_probability += (df['prior_auth'] == 0).astype(float) * 0.30

# Certain procedures have higher denial rates
high_denial_procs = ['27447', '43239', '70553']
denial_probability += df['procedure_code'].isin(high_denial_procs).astype(float) * 0.20

# Emergency room less likely to be denied
denial_probability -= (df['place_of_service'] == 'Emergency Room').astype(float) * 0.15

# Medicare Advantage denies more than FFS
denial_probability += (df['payer_type'] == 'Medicare Advantage').astype(float) * 0.10

# Specialist claims denied more
denial_probability += (df['is_specialist'] == 1).astype(float) * 0.08

# Clip to valid probability range
denial_probability = np.clip(denial_probability, 0.05, 0.85)

# Generate actual denial outcome
df['denied'] = np.random.binomial(1, denial_probability)

# ── Save to CSV ────────────────────────────────────────
df.to_csv('claims.csv', index=False)

print(f"Dataset created with {n:,} claims")
print(f"\nDenial rate: {df['denied'].mean():.1%}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"  {col}")
print(f"\nFirst 3 rows:")
print(df.head(3))