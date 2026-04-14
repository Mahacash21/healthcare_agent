import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────
df = pd.read_csv("claims.csv")

print("=" * 50)
print("CLAIMS DATA EXPLORATION")
print("=" * 50)

# ── Basic shape ────────────────────────────────────────
print(f"\nTotal claims:     {len(df):,}")
print(f"Total columns:    {len(df.columns)}")
print(f"Denied claims:    {df['denied'].sum():,}")
print(f"Approved claims:  {(df['denied']==0).sum():,}")
print(f"Overall denial rate: {df['denied'].mean():.1%}")


# ── Denial rate by provider ────────────────────────────
print("\n--- Denial Rate by Provider Type ---")
prov_denial = df.groupby('provider_type')['denied'].agg(['mean','count'])
prov_denial.columns = ['denial_rate', 'total_claims']
prov_denial['denial_rate'] = prov_denial['denial_rate'].map('{:.1%}'.format)
print(prov_denial.sort_values('denial_rate', ascending=False))

# ── Denial rate by procedure ───────────────────────────
print("\n--- Denial Rate by Procedure Code ---")
proc_denial = df.groupby('procedure_code')['denied'].agg(['mean','count'])
proc_denial.columns = ['denial_rate', 'total_claims']
proc_denial['denial_rate'] = proc_denial['denial_rate'].map('{:.1%}'.format)
print(proc_denial.sort_values('denial_rate', ascending=False))

# ── Prior auth impact ──────────────────────────────────
print("\n--- Impact of Prior Authorization ---")
auth_denial = df.groupby('prior_auth')['denied'].mean()
print(f"  With prior auth:    {auth_denial[1]:.1%} denial rate")
print(f"  Without prior auth: {auth_denial[0]:.1%} denial rate")

# ── Claim amount stats ─────────────────────────────────
print("\n--- Claim Amount Statistics ---")
print(f"  Average claim:  ${df['claim_amount'].mean():,.2f}")
print(f"  Median claim:   ${df['claim_amount'].median():,.2f}")
print(f"  Highest claim:  ${df['claim_amount'].max():,.2f}")
print(f"  Lowest claim:   ${df['claim_amount'].min():,.2f}")

print("\n--- Average Claim Amount: Denied vs Approved ---")
amt = df.groupby('denied')['claim_amount'].mean()
print(f"  Approved claims avg: ${amt[0]:,.2f}")
print(f"  Denied claims avg:   ${amt[1]:,.2f}")

# ── Place of service ───────────────────────────────────
print("\n--- Denial Rate by Place of Service ---")
pos_denial = df.groupby('place_of_service')['denied'].agg(['mean','count'])
pos_denial.columns = ['denial_rate', 'total_claims']
pos_denial['denial_rate'] = pos_denial['denial_rate'].map('{:.1%}'.format)
print(pos_denial.sort_values('denial_rate', ascending=False))

# ── Missing values check ───────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values found")
else:
    print(missing[missing > 0])

print("\n✅ Exploration complete — ready to build the model!")