
import streamlit as st
import pandas as pd
import joblib
import uuid
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FinTrust | Loan Application",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background-color: #f4f7fb;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        color: #12315a;
        letter-spacing: -1px;
    }

    .brand span {
        color: #12a889;
    }

    .hero {
        background: linear-gradient(120deg, #102d52, #1e527c);
        padding: 32px;
        border-radius: 18px;
        color: white;
        margin: 15px 0 24px 0;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #dce8f5;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
        padding: 22px;
        border: 1px solid #e1e8f0;
        border-radius: 15px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #17385e;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #718096;
        margin-bottom: 20px;
    }

    .stTextInput label,
    .stSelectbox label,
    .stNumberInput label {
        color: #243b53;
        font-weight: 600;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #087f6b;
        color: white;
        border: none;
        border-radius: 9px;
        min-height: 48px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #066657;
        color: white;
        border: none;
    }

    .footer {
        text-align: center;
        color: #78879a;
        font-size: 12px;
        padding: 25px 0;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border: 1px solid #e1e8f0;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER
# ==========================================

left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<div class="brand">FinTrust<span>.</span></div>',
        unsafe_allow_html=True
    )
    st.caption("DIGITAL LENDING EXPERIENCE")

with right:
    st.markdown(
        "<div style='text-align:right; padding-top:10px;"
        "color:#58708b;font-weight:600;'>🔒 Application Portal</div>",
        unsafe_allow_html=True
    )

st.markdown("""
<div class="hero">
    <h1>Let's get you closer to your goals.</h1>
    <p>Complete your loan application to receive a preliminary
    machine-learning assessment.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run "
        "`python train_model.py` before opening this application."
    )
    st.stop()

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["pipeline"]
    feature_columns = model_data["feature_columns"]
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ==========================================
# 5. APPLICATION PROGRESS
# ==========================================

st.markdown("### Loan application")
st.caption("APPLICATION FORM  /  PERSONAL & FINANCIAL DETAILS")

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("🟢 **01 — Applicant details**")

with p2:
    st.markdown("🔵 **02 — Loan information**")

with p3:
    st.markdown("⚪ **03 — Assessment**")

st.progress(0.66)

st.info(
    "Please provide accurate information. This educational application "
    "does not connect to a bank or credit bureau."
)

# ==========================================
# 6. APPLICATION FORM
# ==========================================

with st.form("loan_application", clear_on_submit=False):

    # Personal details
    st.markdown(
        '<div class="section-title">👤 Personal information</div>'
        '<div class="section-subtitle">Basic applicant information</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        full_name = st.text_input(
            "Full name",
            placeholder="Enter your full name"
        )

        email = st.text_input(
            "Email address",
            placeholder="you@example.com"
        )

        gender = st.selectbox(
            "Gender (dataset field)",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Marital status",
            ["No", "Yes"]
        )

    with c2:
        phone = st.text_input(
            "Phone number",
            placeholder="Enter your contact number"
        )

        dependents = st.selectbox(
            "Number of dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education level",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Employment category",
            ["No", "Yes"],
            format_func=lambda x: (
                "Salaried / Not self-employed"
                if x == "No"
                else "Self-employed"
            )
        )

    st.divider()

    # Financial details
    st.markdown(
        '<div class="section-title">💰 Financial information</div>'
        '<div class="section-subtitle">Income details used by the '
        'prediction model</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        applicant_income = st.number_input(
            "Applicant income (per month)",
            min_value=0,
            max_value=100000000,
            value=30000,
            step=1000
        )

    with c2:
        coapplicant_income = st.number_input(
            "Co-applicant income (per month)",
            min_value=0,
            max_value=100000000,
            value=0,
            step=1000
        )

    st.caption(
        "Income values are entered in the same numeric scale used by "
        "the practice dataset. They are not independently verified."
    )

    st.divider()

    # Loan details
    st.markdown(
        '<div class="section-title">🏠 Loan requirements</div>'
        '<div class="section-subtitle">Tell us about the loan you need</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        loan_amount = st.number_input(
            "Requested loan amount (dataset units)",
            min_value=1,
            max_value=10000000,
            value=120,
            step=10
        )

        loan_term = st.selectbox(
            "Repayment term",
            [360, 300, 240, 180, 120, 60, 36, 12],
            format_func=lambda x: f"{x} months"
        )

    with c2:
        property_area = st.selectbox(
            "Property location",
            ["Urban", "Semiurban", "Rural"]
        )

        credit_history = st.selectbox(
            "Credit history indicator",
            [1.0, 0.0],
            format_func=lambda x: (
                "Meets historical guidelines"
                if x == 1.0
                else "Does not meet historical guidelines"
            )
        )

    st.caption(
        "Loan amount is entered in the dataset's units, not necessarily "
        "in rupees. The credit-history field is a simplified indicator "
        "from the historical training dataset."
    )

    st.divider()

    consent = st.checkbox(
        "I confirm that the information entered is accurate and "
        "understand this is an educational prediction."
    )

    submitted = st.form_submit_button(
        "Submit application for assessment →",
        use_container_width=True
    )

# ==========================================
# 7. VALIDATION AND PREDICTION
# ==========================================

if submitted:

    errors = []

    if not full_name.strip():
        errors.append("Please enter your full name.")

    if not email.strip() or "@" not in email:
        errors.append("Please enter a valid email address.")

    if not phone.strip():
        errors.append("Please enter your phone number.")

    if not consent:
        errors.append("Please confirm the information statement.")

    if errors:
        for error in errors:
            st.error(error)

    else:
        # Only dataset features are passed to the model.
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }])

        try:
            input_data = input_data[feature_columns]

            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            approved_index = list(model.classes_).index(1)
            probability = float(probabilities[approved_index])

            if "application_reference" not in st.session_state:
                st.session_state.application_reference = (
                    "FT-" + uuid.uuid4().hex[:8].upper()
                )

            st.session_state.last_result = {
                "name": full_name,
                "reference": st.session_state.application_reference,
                "prediction": prediction,
                "probability": probability,
                "submitted_at": datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                )
            }

        except Exception as error:
            st.error(f"Unable to process this application: {error}")

# ==========================================
# 8. RESULTS PAGE
# ==========================================

if "last_result" in st.session_state:

    result = st.session_state.last_result

    st.divider()

    st.markdown("## Application assessment")

    st.caption(
        f"Reference: {result['reference']}  |  "
        f"Submitted: {result['submitted_at']}"
    )

    st.markdown(f"### Thank you, {result['name']}.")

    if result["prediction"] == 1:
        st.success(
            "Preliminary model result: Likely approved"
        )
    else:
        st.warning(
            "Preliminary model result: Likely not approved"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model-estimated approval probability",
            f"{result['probability']:.1%}"
        )

    with col2:
        st.metric(
            "Application reference",
            result["reference"]
        )

    st.progress(result["probability"])

    st.info(
        "This is an educational machine-learning output, not an actual "
        "loan approval or rejection. The model uses historical practice "
        "data and has not been validated for real lending."
    )

    st.markdown("### What happens next?")

    st.markdown("""
    1. **Application assessment:** The model evaluates the entered
       dataset-supported features.
    2. **Review:** In a real bank, authorized staff would verify
       documents, affordability, identity, and applicable requirements.
    3. **Final decision:** A real lender makes its own decision.
    """)

    st.caption(
        "No bank, credit bureau, or external lending service is contacted "
        "by this demo. Contact information is not used by the prediction model."
    )

# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    FinTrust Demo • Loan Prediction Project<br>
    Educational use only • Not a financial institution
</div>
""", unsafe_allow_html=True)
