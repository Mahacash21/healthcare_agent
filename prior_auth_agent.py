import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# ══════════════════════════════════════════════
# STEP 1 — Define the Agent's State
# ══════════════════════════════════════════════
# State is like the agent's memory — it carries
# information from one step to the next

class AgentState(TypedDict):
    # Input fields
    procedure_code:   str
    diagnosis_code:   str
    provider_type:    str
    payer_type:       str
    patient_age:      int
    claim_amount:     float
    num_diagnoses:    int
    prior_auth:       int
    is_specialist:    int

    # Fields the agent fills in as it works
    policy_context:   str    # what the policy says
    denial_risk:      float  # ML model prediction
    decision:         str    # approve / deny / escalate
    response_letter:  str    # drafted PA letter
    messages:         List   # conversation history

# ══════════════════════════════════════════════
# STEP 2 — Load Tools
# ══════════════════════════════════════════════
print("Loading tools...")

# Tool 1 — Vector DB for policy lookup
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "./db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Tool 2 — ML model for denial prediction
with open('claims_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Tool 3 — LLM for reasoning and writing
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("Tools loaded!")

# ══════════════════════════════════════════════
# STEP 3 — Define Agent Nodes (Steps)
# ══════════════════════════════════════════════

def lookup_policy(state: AgentState) -> AgentState:
    """Node 1: Search policy documents for coverage rules"""
    print("  Searching policy documents...")

    query = f"""
    Prior authorization policy for:
    Procedure: {state['procedure_code']}
    Diagnosis: {state['diagnosis_code']}
    Provider: {state['provider_type']}
    Payer: {state['payer_type']}
    """

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    state['policy_context'] = context
    return state


def predict_denial_risk(state: AgentState) -> AgentState:
    """Node 2: Run ML model to predict denial probability"""
    print("  Running denial prediction model...")

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

    text_columns = [
        'procedure_code', 'diagnosis_code', 'provider_type',
        'payer_type', 'place_of_service'
    ]
    for col in text_columns:
        input_data[col] = encoders[col].transform(input_data[col])

    probability = ml_model.predict_proba(input_data)[0]
    state['denial_risk'] = float(probability[1])

    print(f"  Denial risk: {state['denial_risk']:.1%}")
    return state


def make_decision(state: AgentState) -> AgentState:
    """Node 3: Decide to approve, deny, or escalate"""
    print("  Making decision...")

    risk = state['denial_risk']

    if risk < 0.3:
        state['decision'] = 'approve'
    elif risk > 0.7:
        state['decision'] = 'deny'
    else:
        state['decision'] = 'escalate'

    print(f"  Decision: {state['decision'].upper()}")
    return state


def draft_response(state: AgentState) -> AgentState:
    """Node 4: Write the prior auth response letter"""
    print("  Drafting response letter...")

    decision_text = {
        'approve': 'APPROVED',
        'deny':    'DENIED',
        'escalate': 'PENDING MEDICAL REVIEW'
    }[state['decision']]

    prompt = f"""
    You are a healthcare utilization management specialist.
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

    Keep it professional and concise.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    state['response_letter'] = response.content
    return state


def route_decision(state: AgentState) -> str:
    """Conditional Edge: Routes to different paths based on decision"""
    return state['decision']


# ══════════════════════════════════════════════
# STEP 4 — Build the Graph
# ══════════════════════════════════════════════
print("Building agent graph...")

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("lookup_policy",      lookup_policy)
workflow.add_node("predict_denial",     predict_denial_risk)
workflow.add_node("make_decision",      make_decision)
workflow.add_node("draft_response",     draft_response)

# Add edges (connections between nodes)
workflow.set_entry_point("lookup_policy")
workflow.add_edge("lookup_policy",  "predict_denial")
workflow.add_edge("predict_denial", "make_decision")
workflow.add_edge("make_decision",  "draft_response")
workflow.add_edge("draft_response", END)

# Compile the graph
agent = workflow.compile()
print("Agent ready!")

# ══════════════════════════════════════════════
# STEP 5 — Test the Agent
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("TESTING PRIOR AUTH AGENT")
print("="*60)

# Test case 1 — High risk claim
test_request = {
    "procedure_code":   "27447",    # knee replacement
    "diagnosis_code":   "M79.3",
    "provider_type":    "Orthopedics",
    "payer_type":       "Medicare Advantage",
    "patient_age":      68,
    "claim_amount":     8500.00,
    "num_diagnoses":    3,
    "prior_auth":       0,           # no prior auth!
    "is_specialist":    1,
    "policy_context":   "",
    "denial_risk":      0.0,
    "decision":         "",
    "response_letter":  "",
    "messages":         []
}

print("\nProcessing prior auth request...")
print(f"Procedure: {test_request['procedure_code']} (Knee Replacement)")
print(f"Payer: {test_request['payer_type']}")
print(f"Amount: ${test_request['claim_amount']:,.2f}")
print(f"Prior Auth: {'Yes' if test_request['prior_auth'] else 'No'}")
print("\nAgent steps:")

result = agent.invoke(test_request)

print("\n" + "="*60)
print("AGENT RESULT")
print("="*60)
print(f"\nDenial Risk:  {result['denial_risk']:.1%}")
print(f"Decision:     {result['decision'].upper()}")
print(f"\nResponse Letter:")
print("-"*40)
print(result['response_letter'])