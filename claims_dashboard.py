import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Claims Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("claims.csv")
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────
st.sidebar.title("Filters")

selected_payers = st.sidebar.multiselect(
    "Payer Type",
    options=df['payer_type'].unique(),
    default=df['payer_type'].unique()
)

selected_providers = st.sidebar.multiselect(
    "Provider Type",
    options=df['provider_type'].unique(),
    default=df['provider_type'].unique()
)

amount_range = st.sidebar.slider(
    "Claim Amount Range ($)",
    min_value=int(df['claim_amount'].min()),
    max_value=int(df['claim_amount'].max()),
    value=(int(df['claim_amount'].min()),
           int(df['claim_amount'].max()))
)

# ── Apply filters ──────────────────────────────────────
filtered = df[
    (df['payer_type'].isin(selected_payers)) &
    (df['provider_type'].isin(selected_providers)) &
    (df['claim_amount'] >= amount_range[0]) &
    (df['claim_amount'] <= amount_range[1])
]

# ── Page header ────────────────────────────────────────
st.title("📊 Claims Analytics Dashboard")
st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** total claims")
st.divider()

# ══════════════════════════════════════════════════════
# SECTION 1 — KPI METRICS
# ══════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Claims",
        value=f"{len(filtered):,}"
    )

with col2:
    denial_rate = filtered['denied'].mean()
    st.metric(
        label="Denial Rate",
        value=f"{denial_rate:.1%}"
    )

with col3:
    avg_amount = filtered['claim_amount'].mean()
    st.metric(
        label="Avg Claim Amount",
        value=f"${avg_amount:,.2f}"
    )

with col4:
    total_spend = filtered['claim_amount'].sum()
    st.metric(
        label="Total Spend",
        value=f"${total_spend:,.0f}"
    )

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 2 — DENIAL RATE BY PAYER AND PROCEDURE
# ══════════════════════════════════════════════════════
col5, col6 = st.columns(2)

with col5:
    st.subheader("Denial Rate by Payer")

    payer_denial = (
        filtered.groupby('payer_type')['denied']
        .mean()
        .reset_index()
        .rename(columns={'denied': 'denial_rate'})
        .sort_values('denial_rate', ascending=False)
    )

    fig1 = px.bar(
        payer_denial,
        x='payer_type',
        y='denial_rate',
        color='denial_rate',
        color_continuous_scale='Reds',
        labels={'payer_type': 'Payer', 'denial_rate': 'Denial Rate'},
        text=payer_denial['denial_rate'].map('{:.1%}'.format)
    )
    fig1.update_layout(
        yaxis_tickformat='.0%',
        showlegend=False,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with col6:
    st.subheader("Denial Rate by Procedure Code")

    proc_denial = (
        filtered.groupby('procedure_code')['denied']
        .mean()
        .reset_index()
        .rename(columns={'denied': 'denial_rate'})
        .sort_values('denial_rate', ascending=False)
    )

    fig2 = px.bar(
        proc_denial,
        x='procedure_code',
        y='denial_rate',
        color='denial_rate',
        color_continuous_scale='Oranges',
        labels={'procedure_code': 'CPT Code', 'denial_rate': 'Denial Rate'},
        text=proc_denial['denial_rate'].map('{:.1%}'.format)
    )
    fig2.update_layout(
        yaxis_tickformat='.0%',
        showlegend=False,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 3 — AMOUNT VS DENIAL RATE SCATTER
# ══════════════════════════════════════════════════════
st.subheader("Claim Amount vs Denial Rate by Provider Type")

scatter_data = (
    filtered.groupby('provider_type')
    .agg(
        denial_rate=('denied', 'mean'),
        avg_amount=('claim_amount', 'mean'),
        total_claims=('denied', 'count')
    )
    .reset_index()
)

fig3 = px.scatter(
    scatter_data,
    x='avg_amount',
    y='denial_rate',
    size='total_claims',
    color='denial_rate',
    color_continuous_scale='Reds',
    hover_name='provider_type',
    labels={
        'avg_amount':   'Average Claim Amount ($)',
        'denial_rate':  'Denial Rate',
        'total_claims': 'Total Claims'
    },
    text='provider_type'
)
fig3.update_traces(textposition='top center')
fig3.update_layout(
    yaxis_tickformat='.0%',
    coloraxis_showscale=False
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 4 — PRIOR AUTH IMPACT
# ══════════════════════════════════════════════════════
col7, col8 = st.columns(2)

with col7:
    st.subheader("Prior Auth Impact on Denials")

    auth_data = (
        filtered.groupby('prior_auth')['denied']
        .mean()
        .reset_index()
    )
    auth_data['prior_auth'] = auth_data['prior_auth'].map(
        {1: 'With Prior Auth', 0: 'Without Prior Auth'}
    )
    auth_data.rename(columns={'denied': 'denial_rate'}, inplace=True)

    fig4 = px.bar(
        auth_data,
        x='prior_auth',
        y='denial_rate',
        color='prior_auth',
        color_discrete_map={
            'With Prior Auth':    '#2ecc71',
            'Without Prior Auth': '#e74c3c'
        },
        labels={
            'prior_auth':   'Prior Authorization',
            'denial_rate':  'Denial Rate'
        },
        text=auth_data['denial_rate'].map('{:.1%}'.format)
    )
    fig4.update_layout(
        yaxis_tickformat='.0%',
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

with col8:
    st.subheader("Denial Rate by Place of Service")

    pos_data = (
        filtered.groupby('place_of_service')['denied']
        .mean()
        .reset_index()
        .rename(columns={'denied': 'denial_rate'})
        .sort_values('denial_rate', ascending=True)
    )

    fig5 = px.bar(
        pos_data,
        x='denial_rate',
        y='place_of_service',
        orientation='h',
        color='denial_rate',
        color_continuous_scale='Blues',
        labels={
            'place_of_service': 'Place of Service',
            'denial_rate':      'Denial Rate'
        },
        text=pos_data['denial_rate'].map('{:.1%}'.format)
    )
    fig5.update_layout(
        xaxis_tickformat='.0%',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 5 — RAW DATA TABLE
# ══════════════════════════════════════════════════════
st.subheader("Raw Claims Data")

col9, col10 = st.columns([3, 1])
with col9:
    search = st.text_input("Search by procedure code or payer")
with col10:
    show_denied_only = st.checkbox("Show denied claims only")

display_df = filtered.copy()

if search:
    display_df = display_df[
        display_df['procedure_code'].str.contains(search, case=False) |
        display_df['payer_type'].str.contains(search, case=False)
    ]

if show_denied_only:
    display_df = display_df[display_df['denied'] == 1]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(display_df):,} claims")

# ── Download button ────────────────────────────────────
csv = display_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_claims.csv",
    mime="text/csv"
)