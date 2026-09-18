# OVB Simulator

### Overview
This app simulates Omitted Variable Bias (OVB). It generates a treatment $X$, a confounder $Z$, and an outcome $Y$ where $Y = \beta_1 X + \beta_2 Z + \epsilon$. 

### Key Concept
The **Naive Model** (omitting $Z$) yields a biased estimate of $\beta_1$ because $X$ and $Z$ are correlated. The bias equals $\beta_2 \times \text{Corr}(X, Z)$.

### Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`

**Citation:** This application was developed with the assistance of **Google Gemini AI Studio**.
