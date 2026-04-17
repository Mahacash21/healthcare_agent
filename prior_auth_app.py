import os
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

st.set_page_config(
    page_title="Prior Auth AI Agent",
    page_icon="🏥",
    layout="wide"
)

# ── Load all tools once ────────────────────────────────
@st.cache_resource
def load_agent():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "./db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    with open('claims_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # ── Agent State ────────────────────────────────────
    class AgentState(TypedDict):
        procedure_code:  str
        diagnosis_code:  str
        provider_type:   str
        payer_type:      str
        patient_age:     int
        claim_amount:    float
        num_diagnoses:   int
        prior_auth:      int
        is_specialist:   int
        policy_context:  str
        denial_risk:     float
        decision:        str
        response_letter: str
        messages:        List

    # ── Node functions ─────────────────────────────────
    def lookup_policy(state):
        query = f"""Prior authorization for procedure {state['procedure_code']}
        diagnosis {state['diagnosis_code']} payer {state['payer_type']}"""
        docs = vectorstore.similarity_search(query, k=3)
        state['policy_context'] = "\n\n".join([d.page_content for d in docs])
        return state

    def predict_denial(state):
        input_data = pd.DataFrame([{
            'procedure_code':   state['procedure_code'],
            'diagnosis_code':   state['diagnosis_code'],
            'provider_type':    state['provider_type'],
            'payer_type':       state['payer_type'],
            'place_of_service': 'Outpatient Hospital',
            'patient_age':      state['patient_age'],
            'claim_amount':     state['claim_amount'],
            'num_diagnoses':    state['num_diagnoses'],
            'prior_auth':       state['prior_auth'],
            'is_specialist':    state['is_specialist']
        }])
        text_cols = ['procedure_code','diagnosis_code','provider_type',
                     'payer_type','place_of_service']
        for col in text_cols:
            input_data[col] = encoders[col].transform(input_data[col])
        prob = ml_model.predict_proba(input_data)[0]
        state['denial_risk'] = float(prob[1])
        return state

    def make_decision(state):
        risk = state['denial_risk']
        if risk < 0.3:
            state['decision'] = 'approve'
        elif risk > 0.7:
            state['decision'] = 'deny'
        else:
            state['decision'] = 'escalate'
        return state

    def draft_response(state):
        decision_text = {
            'approve':  'APPROVED',
            'deny':     'DENIED',
            'escalate': 'PENDING MEDICAL REVIEW'
        }[state['decision']]

        prompt = f"""You are a healthcare utilization management specialist.
Write a professional prior authorization determination letter.

Claim Details:
- Procedure: {state['procedure_code']}
- Diagnosis: {state['diagnosis_code']}
- Provider: {state['provider_type']}
- Payer: {state['payer_type']}
- Patient Age: {state['patient_age']}
- Claim Amount: ${state['claim_amount']:,.2f}
- Denial Risk Score: {state['denial_risk']:.1%}

Relevant Policy Context:
{state['policy_context'][:500]}

Determination: {decision_text}

Write a formal 3-paragraph letter explaining:
1. The determination decision
2. The clinical and policy rationale
3. Next steps or appeal rights

Keep it professional and concise."""

        response = llm.invoke([HumanMessage(content=prompt)])
        state['response_letter'] = response.content
        return state

    # ── Build graph ────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("lookup_policy",  lookup_policy)
    workflow.add_node("predict_denial", predict_denial)
    workflow.add_node("make_decision",  make_decision)
    workflow.add_node("draft_response", draft_response)
    workflow.set_entry_point("lookup_policy")
    workflow.add_edge("lookup_policy",  "predict_denial")
    workflow.add_edge("predict_denial", "make_decision")
    workflow.add_edge("make_decision",  "draft_response")
    workflow.add_edge("draft_response", END)

    return workflow.compile()

agent = load_agent()

# ── Page Header ────────────────────────────────────────
st.title("🏥 Prior Authorization AI Agent")
st.markdown("Autonomous clinical decision support powered by AI")
st.divider()

# ── Input Form ─────────────────────────────────────────
st.subheader("Prior Auth Request Details")

col1, col2, col3 = st.columns(3)

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

with col2:
    payer_type = st.selectbox("Payer Type", [
        'Medicare Advantage', 'Commercial', 'Medicaid', 'Medicare FFS'
    ])
    patient_age = st.slider("Patient Age", 18, 90, 65)
    claim_amount = st.number_input(
        "Claim Amount ($)", min_value=50.0,
        max_value=50000.0, value=1500.0, step=100.0
    )

with col3:
    num_diagnoses = st.slider("Number of Diagnoses", 1, 7, 2)
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

# ── Run Agent ──────────────────────────────────────────
if st.button("Run Prior Auth Agent", type="primary", use_container_width=True):

    with st.spinner("Agent processing request..."):

        # Show live agent steps
        step_container = st.empty()

        steps = [
            "Step 1 of 4 — Searching policy documents...",
            "Step 2 of 4 — Running denial prediction model...",
            "Step 3 of 4 — Making determination decision...",
            "Step 4 of 4 — Drafting response letter...",
        ]

        import time
        for step in steps:
            step_container.info(step)
            time.sleep(0.5)

        # Run the agent
        result = agent.invoke({
            "procedure_code":  procedure_code,
            "diagnosis_code":  diagnosis_code,
            "provider_type":   provider_type,
            "payer_type":      payer_type,
            "patient_age":     patient_age,
            "claim_amount":    claim_amount,
            "num_diagnoses":   num_diagnoses,
            "prior_auth":      prior_auth,
            "is_specialist":   is_specialist,
            "policy_context":  "",
            "denial_risk":     0.0,
            "decision":        "",
            "response_letter": "",
            "messages":        []
        })

        step_container.empty()

    # ── Show Results ───────────────────────────────────
    st.subheader("Agent Determination")

    # Decision banner
    decision = result['decision']
    risk = result['denial_risk']

    if decision == 'approve':
        st.success(f"APPROVED — Denial risk: {risk:.1%}")
    elif decision == 'deny':
        st.error(f"DENIED — Denial risk: {risk:.1%}")
    else:
        st.warning(f"PENDING MEDICAL REVIEW — Denial risk: {risk:.1%}")

    # Metrics row
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        st.metric("Denial Risk",    f"{risk:.1%}")
    with col5:
        st.metric("Approval Odds",  f"{1-risk:.1%}")
    with col6:
        st.metric("Claim Amount",   f"${claim_amount:,.0f}")
    with col7:
        st.metric("Decision",       decision.upper())

    # Risk factors
    st.divider()
    col8, col9 = st.columns(2)

    with col8:
        st.subheader("Risk Factors Detected")
        factors = []
        if prior_auth == 0:
            factors.append("No prior authorization obtained")
        if claim_amount > 5000:
            factors.append(f"High claim amount (${claim_amount:,.0f})")
        if payer_type == "Medicare Advantage":
            factors.append("Medicare Advantage — higher denial rate")
        if procedure_code in ['27447', '43239', '70553']:
            factors.append(f"Procedure {procedure_code} — high denial history")
        if is_specialist:
            factors.append("Specialist visit — elevated risk")

        if factors:
            for f in factors:
                st.warning(f)
        else:
            st.success("No major risk factors detected")

    with col9:
        st.subheader("Policy Context Found")
        with st.expander("View relevant policy excerpts"):
            st.caption(result['policy_context'][:600] + "...")

    # Response letter
    st.divider()
    st.subheader("Generated Determination Letter")
    st.markdown(result['response_letter'])

    # Download button
    st.download_button(
        label="Download Letter as Text File",
        data=result['response_letter'],
        file_name=f"prior_auth_{procedure_code}_{decision}.txt",
        mime="text/plain"
    )