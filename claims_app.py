import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Claims Denial Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Load model and encoders ────────────────────────────
@st.cache_resource
def load_model():
    with open('claims_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# ── Page header ────────────────────────────────────────
st.title("🏥 Claims Denial Predictor")
st.markdown("Enter claim details below to predict denial risk.")
st.divider()

# ── Input form ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    procedure_code = st.selectbox("Procedure Code (CPT)", [
        '99213', '99214', '99232', '93000', '71046',
        '80053', '99283', '27447', '43239', '70553'
    ])

    diagnosis_code = st.selectbox("Diagnosis Code (ICD-10)", [
        'I10', 'E11.9', 'J18.9', 'M79.3', 'Z12.11',
        'F32.9', 'K21.0', 'N39.0', 'J06.9', 'R51'
    ])

    provider_type = st.selectbox("Provider Type", [
        'Internal Medicine', 'Family Practice', 'Cardiology',
        'Orthopedics', 'Emergency Medicine', 'Radiology',
        'Gastroenterology', 'Psychiatry', 'Urology', 'Neurology'
    ])

    payer_type = st.selectbox("Payer Type", [
        'Medicare Advantage', 'Commercial', 'Medicaid', 'Medicare FFS'
    ])

    place_of_service = st.selectbox("Place of Service", [
        'Office', 'Outpatient Hospital', 'Emergency Room',
        'Inpatient Hospital', 'Telehealth'
    ])

with col2:
    patient_age = st.slider(
        "Patient Age",
        min_value=18,
        max_value=90,
        value=65
    )

    claim_amount = st.number_input(
        "Claim Amount ($)",
        min_value=50.0,
        max_value=50000.0,
        value=1500.0,
        step=100.0
    )

    num_diagnoses = st.slider(
        "Number of Diagnoses",
        min_value=1,
        max_value=7,
        value=2
    )

    prior_auth = st.radio(
        "Prior Authorization Obtained?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True
    )

    is_specialist = st.radio(
        "Specialist Visit?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True
    )

st.divider()

# ── Predict button ─────────────────────────────────────
if st.button("Predict Denial Risk", type="primary", use_container_width=True):

    # Build input dataframe
    input_data = pd.DataFrame([{
        'procedure_code':   procedure_code,
        'diagnosis_code':   diagnosis_code,
        'provider_type':    provider_type,
        'payer_type':       payer_type,
        'place_of_service': place_of_service,
        'patient_age':      patient_age,
        'claim_amount':     claim_amount,
        'num_diagnoses':    num_diagnoses,
        'prior_auth':       prior_auth,
        'is_specialist':    is_specialist
    }])

    # Encode text columns
    text_columns = [
        'procedure_code', 'diagnosis_code', 'provider_type',
        'payer_type', 'place_of_service'
    ]
    for col in text_columns:
        input_data[col] = encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    denial_risk = probability[1]

    # ── Show results ───────────────────────────────────
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"LIKELY DENIED — {denial_risk:.1%} denial risk")
    else:
        st.success(f"LIKELY APPROVED — {denial_risk:.1%} denial risk")

    # Risk gauge
    st.markdown("**Denial Risk Score**")
    st.progress(float(denial_risk))

    # Breakdown
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Denial Risk",    f"{denial_risk:.1%}")
    with col4:
        st.metric("Approval Odds",  f"{probability[0]:.1%}")
    with col5:
        st.metric("Claim Amount",   f"${claim_amount:,.0f}")

    # Key risk factors
    st.divider()
    st.subheader("Key Risk Factors")

    risk_factors = []
    if prior_auth == 0:
        risk_factors.append("No prior authorization obtained")
    if claim_amount > 5000:
        risk_factors.append(f"High claim amount (${claim_amount:,.0f})")
    if payer_type == "Medicare Advantage":
        risk_factors.append("Medicare Advantage has higher denial rates")
    if procedure_code in ['27447', '43239', '70553']:
        risk_factors.append(f"Procedure {procedure_code} has high denial history")
    if is_specialist == 1:
        risk_factors.append("Specialist visits have higher denial rates")

    if risk_factors:
        for factor in risk_factors:
            st.warning(factor)
    else:
        st.info("No major risk factors detected for this claim")