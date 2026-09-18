import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Omitted Variable Bias Demo", layout="wide")

st.title("📉 Omitted Variable Bias (OVB) Simulator")
st.markdown("""
This app demonstrates how excluding a confounder ($Z$) from your regression leads to a biased estimate of the effect of $X$ on $Y$.
The True Model is: $Y = \\beta_1 X + \\beta_2 Z + \\epsilon$
""")

# --- Sidebar Controls ---
st.sidebar.header("Parameters")

true_beta1 = 1.0
st.sidebar.markdown(f"**True $\\beta_1$ (Effect of X):** {true_beta1}")

# Helper functions to sync slider and number input
def update_slider(key_prefix):
    st.session_state[f"{key_prefix}_slider"] = st.session_state[f"{key_prefix}_input"]

def update_input(key_prefix):
    st.session_state[f"{key_prefix}_input"] = st.session_state[f"{key_prefix}_slider"]

# 1. Confounder Strength (beta2)
st.sidebar.subheader("Confounder Strength (β2)")
beta2 = st.sidebar.number_input(
    "Type value:", min_value=-5.0, max_value=5.0, value=2.0, step=0.1, 
    key="beta2_input", on_change=update_slider, args=("beta2",)
)
st.sidebar.slider(
    "Slide to adjust:", min_value=-5.0, max_value=5.0, value=2.0, step=0.1, 
    key="beta2_slider", on_change=update_input, args=("beta2",), label_visibility="collapsed"
)

# 2. Correlation (rho)
st.sidebar.subheader("Correlation (ρ) X & Z")
rho = st.sidebar.number_input(
    "Type value:", min_value=-0.9, max_value=0.9, value=0.5, step=0.05, 
    key="rho_input", on_change=update_slider, args=("rho",)
)
st.sidebar.slider(
    "Slide to adjust:", min_value=-0.9, max_value=0.9, value=0.5, step=0.05, 
    key="rho_slider", on_change=update_input, args=("rho",), label_visibility="collapsed"
)

# 3. Sample Size (N)
st.sidebar.subheader("Sample Size (N)")
n_samples = st.sidebar.number_input(
    "Type value:", min_value=50, max_value=5000, value=500, step=50, 
    key="n_input", on_change=update_slider, args=("n",)
)
st.sidebar.slider(
    "Slide to adjust:", min_value=50, max_value=5000, value=500, step=50, 
    key="n_slider", on_change=update_input, args=("n",), label_visibility="collapsed"
)

st.sidebar.info(
    "**Note:** Increasing sample size makes the estimates more 'precise' (smaller standard errors), "
    "but it **does not** reduce the bias in the Naive Model. Bias is a structural issue, not a sampling issue."
)

# --- Data Generation ---
np.random.seed(42)
X = np.random.normal(0, 1, n_samples)
Z_noise = np.random.normal(0, 1, n_samples)
Z = rho * X + np.sqrt(1 - rho**2) * Z_noise
epsilon = np.random.normal(0, 0.5, n_samples)
Y = (true_beta1 * X) + (beta2 * Z) + epsilon
df = pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

# --- Regression Analysis ---
X_full = sm.add_constant(df[['X', 'Z']])
model_full = sm.OLS(df['Y'], X_full).fit()
beta1_hat_full = model_full.params['X']

X_naive = sm.add_constant(df['X'])
model_naive = sm.OLS(df['Y'], X_naive).fit()
beta1_hat_naive = model_naive.params['X']

observed_bias = beta1_hat_naive - true_beta1
predicted_bias = beta2 * rho

# --- Display Metrics ---
st.subheader("Comparison of Estimates")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("True β1", f"{true_beta1:.2f}")
col2.metric("Full Model $\hat{β}_1$", f"{beta1_hat_full:.3f}")
col3.metric("Naive Model $\hat{β}_1$", f"{beta1_hat_naive:.3f}", f"{observed_bias:.3f}", delta_color="inverse")
col4.metric("Observed Bias", f"{observed_bias:.3f}")
col5.metric("Predicted Bias (β2 * ρ)", f"{predicted_bias:.3f}")

# --- Visualization ---
st.subheader("Visualizing the Bias")
x_range = np.linspace(df['X'].min(), df['X'].max(), 100)
y_full_line = model_full.params['const'] + (beta1_hat_full * x_range)
y_naive_line = model_naive.params['const'] + (beta1_hat_naive * x_range)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='markers', marker=dict(color='gray', opacity=0.3), name='Data Points'))
fig.add_trace(go.Scatter(x=x_range, y=y_full_line, mode='lines', line=dict(color='blue', width=4), name='Full Model (Correct)'))
fig.add_trace(go.Scatter(x=x_range, y=y_naive_line, mode='lines', line=dict(color='red', width=4, dash='dash'), name='Naive Model (Biased)'))

fig.update_layout(xaxis_title="Treatment (X)", yaxis_title="Outcome (Y)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=600)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(f"""
### Why is this happening?
In the **Naive Model**, X "steals" the credit for the effect that Z is actually having on Y, because X and Z are correlated. 
The mathematical bias is:  
$$\\text{{Bias}} = \\beta_2 \\times \\rho$$
""")