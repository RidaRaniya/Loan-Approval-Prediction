
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="FinTrust | Loan Approval",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

MODEL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


# =========================================================
# 2. CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --blue: #2563EB;
        --blue-dark: #1D4ED8;
        --purple: #7C3AED;
        --pink: #EC4899;
        --ink: #172033;
        --muted: #64748B;
        --border: #E3E8F4;
        --page: #F6F7FF;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, #EEF2FF 0, transparent 30%),
            radial-gradient(circle at 95% 8%, #FCE7F3 0, transparent 25%),
            var(--page);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }

    .block-container {
        max-width: 1260px;
        padding-top: .85rem;
        padding-bottom: 1.25rem;
    }

    .topbar {
        background: linear-gradient(
            115deg,
            #2563EB 0%,
            #6D28D9 58%,
            #A21CAF 100%
        );
        color: white;
        border-radius: 20px;
        padding: 20px 25px;
        margin-bottom: 19px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
        box-shadow: 0 12px 28px rgba(79, 70, 229, .22);
        position: relative;
        overflow: hidden;
    }

    .topbar:after {
        content: "";
        position: absolute;
        width: 190px;
        height: 190px;
        border-radius: 50%;
        right: 22%;
        top: -115px;
        background: rgba(255,255,255,.10);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 13px;
        position: relative;
        z-index: 1;
    }

    .brand-icon {
        font-size: 30px;
        width: 52px;
        height: 52px;
        display: grid;
        place-items: center;
        background: rgba(255,255,255,.17);
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 15px;
    }

    .brand-name {
        color: white;
        font-size: 24px;
        font-weight: 850;
        letter-spacing: -.5px;
    }

    .brand-caption {
        color: #EDE9FE;
        font-size: 10px;
        letter-spacing: 1.25px;
        margin-top: 3px;
    }

    .topbar-info {
        color: #F5F3FF;
        text-align: right;
        font-size: 12px;
        position: relative;
        z-index: 1;
    }

    .page-title {
        color: #312E81;
        font-size: 28px;
        font-weight: 850;
        letter-spacing: -.7px;
        margin: 3px 0;
    }

    .page-subtitle {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 17px;
    }

    .section-heading {
        color: #312E81;
        font-size: 16px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #EEF2FF,
            #F5F3FF 65%,
            #FDF2F8
        );
        border: 1px solid #E0E7FF;
        border-left: 4px solid #7C3AED;
        border-radius: 11px;
        padding: 10px 13px;
        margin: 8px 0 13px 0;
    }

    .section-help {
        color: var(--muted);
        font-size: 12px;
        margin: -5px 0 12px 0;
    }

    label,
    [data-testid="stWidgetLabel"] p {
        color: #27324A !important;
        font-weight: 650 !important;
    }

    input,
    textarea {
        color: #172033 !important;
    }

    [data-baseweb="select"] * {
        color: #172033;
    }

    [data-testid="stMetricValue"] {
        color: #4338CA;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
    }

    [data-testid="stForm"] {
        background: rgba(255,255,255,.94);
        border: 1px solid #E0E7FF;
        border-radius: 18px;
        padding: 17px 19px 13px 19px;
        box-shadow: 0 7px 22px rgba(49,46,129,.06);
    }

    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: #D8DEEF !important;
    }

    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button {
        background: linear-gradient(
            100deg,
            #2563EB,
            #7C3AED 72%,
            #A21CAF
        );
        color: white !important;
        border: 0;
        border-radius: 11px;
        min-height: 45px;
        font-weight: 800;
        box-shadow: 0 6px 14px rgba(99,102,241,.22);
        transition: transform .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stButton > button:hover {
        color: white !important;
        box-shadow: 0 9px 18px rgba(99,102,241,.30);
        transform: translateY(-1px);
    }

    .result-approved,
    .result-rejected,
    .result-neutral {
        border-radius: 14px;
        padding: 17px 19px;
        margin: 8px 0 12px 0;
    }

    .result-approved {
        background: linear-gradient(110deg, #ECFDF5, #D1FAE5);
        border: 1px solid #A7F3D0;
        color: #065F46;
    }

    .result-rejected {
        background: linear-gradient(110deg, #FFF1F2, #FFE4E6);
        border: 1px solid #FECDD3;
        color: #9F1239;
    }

    .result-neutral {
        background: linear-gradient(110deg, #EEF2FF, #F3E8FF);
        border: 1px solid #C7D2FE;
        color: #3730A3;
    }

    .result-title {
        font-size: 21px;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .small-note {
        font-size: 12px;
        color: var(--muted);
    }

    @media (max-width: 700px) {
        .topbar {
            padding: 14px;
        }

        .brand-name {
            font-size: 21px;
        }

        .page-title {
            font-size: 24px;
        }
    }

    /* Input box text color — white */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Dropdown selected text — white */
    [data-baseweb="select"] [role="combobox"] {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HEADER
# =========================================================
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-icon">🏦</div>
        <div>
          <div class="brand-name">FinTrust</div>
          <div class="brand-caption">
            LOAN APPLICATION &amp; ELIGIBILITY PORTAL
          </div>
        </div>
      </div>

      <div class="topbar-info">
        <strong>Loan Application Workspace</strong><br>
        Loan Approval Prediction System
      </div>
    </div>

    <div class="page-title">Loan Application</div>

    <div class="page-subtitle">
      Complete the applicant, employment, financial, credit and loan
      details to generate a model estimate.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path: str):
    if not os.path.isfile(model_path):
        return None, None, None, f"Model file not found: {model_path}"

    try:
        saved = joblib.load(model_path)
    except Exception as exc:
        return None, None, None, f"Could not load model file: {exc}"

    if isinstance(saved, dict):
        model = saved.get("pipeline")

        if model is None:
            model = saved.get("model")

        if model is None:
            model = saved.get("classifier")

        features = saved.get("feature_columns")

        model_name = saved.get(
            "model_name",
            type(model).__name__ if model is not None else "Loan Model"
        )
    else:
        model = saved
        features = None
        model_name = type(model).__name__

    if model is None:
        return (
            None,
            None,
            None,
            "The saved file does not contain a model under "
            "pipeline/model/classifier."
        )

    if not features:
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)

        elif hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    features = list(step.feature_names_in_)
                    break

    if not features:
        features = MODEL_FEATURES

    return model, list(features), model_name, None


model, saved_features, model_name, model_error = load_model(
    str(MODEL_PATH)
)

if model_error:
    st.error(model_error)

    st.info(
        "Train or copy your model to the path shown above. "
        "The expected project structure is "
        "`app.py` and `models/loan_model.pkl`."
    )

    st.stop()


# =========================================================
# 5. APPLICATION FORM
# =========================================================
with st.form("loan_application_form"):

    # PERSONAL DETAILS
    st.markdown(
        '<div class="section-heading">1. Personal Details</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        full_name = st.text_input(
            "Full Name *",
            placeholder="Applicant full name"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25,
            step=1
        )

        gender = st.selectbox(
            "Gender *",
            ["Male", "Female"]
        )

    with p2:
        email = st.text_input(
            "Email Address",
            placeholder="name@example.com"
        )

        phone = st.text_input(
            "Phone Number",
            placeholder="Contact number"
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Widowed"]
        )

    with p3:
        dependents = st.selectbox(
            "Number of Dependents *",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education *",
            ["Graduate", "Not Graduate"]
        )

        residential_status = st.selectbox(
            "Residential Status",
            ["Resident", "Non-Resident"]
        )

    address = st.text_input(
        "Residential Address",
        placeholder="House, street, locality"
    )

    # EMPLOYMENT DETAILS
    st.markdown(
        '<div class="section-heading">2. Employment Details</div>',
        unsafe_allow_html=True
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-Employed", "Unemployed", "Retired"]
        )

        employer_name = st.text_input(
            "Employer / Business Name"
        )

    with e2:
        occupation = st.selectbox(
            "Occupation",
            [
                "Private Sector",
                "Government",
                "Business Owner",
                "Professional",
                "Other"
            ]
        )

        work_experience = st.number_input(
            "Total Work Experience (years)",
            min_value=0,
            max_value=60,
            value=2
        )

    with e3:
        current_job_years = st.number_input(
            "Years with Current Employer",
            min_value=0,
            max_value=60,
            value=1
        )

        self_employed = st.selectbox(
            "Self-Employed? *",
            ["No", "Yes"]
        )

    # FINANCIAL DETAILS
    st.markdown(
        '<div class="section-heading">3. Financial Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-help">'
        'Use the same income units used to train your model. '
        'For the standard Loan Prediction dataset, income is typically '
        'recorded as raw numeric amounts.'
        '</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        applicant_income = st.number_input(
            "Applicant Income (₹) *",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )

        coapplicant_income = st.number_input(
            "Co-Applicant Income (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )

    with f2:
        existing_emi = st.number_input(
            "Existing Monthly EMI (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

        other_obligations = st.number_input(
            "Other Monthly Obligations (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with f3:
        savings = st.number_input(
            "Savings / Liquid Assets (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )

        other_income = st.number_input(
            "Other Monthly Income (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )

    # CREDIT PROFILE
    st.markdown(
        '<div class="section-heading">4. Credit Profile</div>',
        unsafe_allow_html=True
    )

    cr1, cr2, cr3 = st.columns(3)

    with cr1:
        credit_history = st.selectbox(
            "Credit History *",
            ["Good / Available", "Poor", "Not Available"]
        )

        credit_score = st.number_input(
            "Credit Score (if known)",
            min_value=0,
            max_value=900,
            value=0,
            step=1
        )

    with cr2:
        existing_loans = st.selectbox(
            "Existing Loans",
            [
                "None",
                "Personal Loan",
                "Vehicle Loan",
                "Home Loan",
                "Multiple Loans"
            ]
        )

        repayment_history = st.selectbox(
            "Past Repayment Record",
            [
                "Not Provided",
                "On Time",
                "Some Late Payments",
                "Defaults"
            ]
        )

    with cr3:
        bank_account = st.selectbox(
            "Bank Account",
            ["Savings", "Current", "Salary", "Other"]
        )

        bank_relationship_years = st.number_input(
            "Banking Relationship (years)",
            min_value=0,
            max_value=80,
            value=1
        )

    # LOAN AND PROPERTY DETAILS
    st.markdown(
        '<div class="section-heading">'
        '5. Loan &amp; Property Details</div>',
        unsafe_allow_html=True
    )

    l1, l2, l3 = st.columns(3)

    with l1:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Home Purchase",
                "Construction",
                "Education",
                "Vehicle",
                "Personal",
                "Business",
                "Other"
            ]
        )

        loan_amount_rupees = st.number_input(
            "Requested Loan Amount (₹) *",
            min_value=0.0,
            value=500000.0,
            step=25000.0
        )

    with l2:
        loan_term_months = st.selectbox(
            "Loan Term (months) *",
            [12, 36, 60, 84, 120, 180, 240, 300, 360],
            index=8
        )

        property_area = st.selectbox(
            "Property Area *",
            ["Urban", "Semiurban", "Rural"]
        )

    with l3:
        property_value = st.number_input(
            "Estimated Property Value (₹)",
            min_value=0.0,
            value=0.0,
            step=100000.0
        )

        down_payment = st.number_input(
            "Down Payment (₹)",
            min_value=0.0,
            value=0.0,
            step=25000.0
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        city = st.text_input("City")

    with c2:
        state = st.text_input("State")

    with c3:
        postal_code = st.text_input("Postal Code")

    consent = st.checkbox(
        "I confirm that the information entered above is accurate."
    )

    submitted = st.form_submit_button(
        "Predict Loan Decision",
        use_container_width=True
    )


# =========================================================
# 6. MODEL INPUT AND PREDICTION HELPERS
# =========================================================
def normalize_prediction(prediction):
    value = str(prediction).strip().lower()

    if value in {"y", "yes", "approved", "approve", "1", "true"}:
        return "Approved"

    if value in {"n", "no", "rejected", "reject", "0", "false"}:
        return "Rejected"

    return str(prediction)


def build_model_input(features):
    # Use the exact feature names expected by the trained model.
    # The standard Kaggle dataset stores LoanAmount in thousands.
    values = {
        "Gender": gender,
        "Married": "Yes" if marital_status == "Married" else "No",
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount_rupees / 1000.0,
        "Loan_Amount_Term": loan_term_months,
        "Credit_History": (
            1.0 if credit_history == "Good / Available"
            else 0.0 if credit_history == "Poor"
            else np.nan
        ),
        "Property_Area": property_area,
    }

    missing = [
        feature for feature in features
        if feature not in values
    ]

    if missing:
        raise ValueError(
            "The model expects feature(s) that this app does not "
            "currently map: "
            + ", ".join(missing)
            + ". Update build_model_input() to match your training columns."
        )

    return pd.DataFrame(
        [{feature: values[feature] for feature in features}],
        columns=features
    )


# =========================================================
# 7. HANDLE SUBMISSION
# =========================================================
if submitted:

    if not full_name.strip():
        st.error("Please enter the applicant's full name.")

    elif applicant_income <= 0:
        st.error("Applicant income must be greater than zero.")

    elif loan_amount_rupees <= 0:
        st.error("Requested loan amount must be greater than zero.")

    elif down_payment > loan_amount_rupees:
        st.error(
            "Down payment cannot be greater than the requested loan amount."
        )

    elif not consent:
        st.warning(
            "Please confirm that the information entered is accurate."
        )

    else:
        try:
            input_df = build_model_input(saved_features)

            prediction = model.predict(input_df)[0]
            decision = normalize_prediction(prediction)

            confidence = None

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    confidence = float(np.max(probabilities)) * 100
                except Exception:
                    confidence = None

            st.markdown("---")

            st.markdown(
                '<div class="section-heading">Prediction Result</div>',
                unsafe_allow_html=True
            )

            if decision == "Approved":
                st.markdown(
                    '<div class="result-approved">'
                    '<div class="result-title">'
                    '✓ Predicted as Approved</div>'
                    'The model predicts an approval outcome based on '
                    'the features used during training.</div>',
                    unsafe_allow_html=True
                )

            elif decision == "Rejected":
                st.markdown(
                    '<div class="result-rejected">'
                    '<div class="result-title">'
                    '✕ Predicted as Rejected</div>'
                    'The model predicts a rejection outcome based on '
                    'the features used during training.</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f'<div class="result-neutral">'
                    f'<div class="result-title">'
                    f'Model Output: {decision}</div>'
                    'The returned label was not recognized as a '
                    'standard approval/rejection label.</div>',
                    unsafe_allow_html=True
                )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Applicant", full_name.strip())

            with m2:
                st.metric(
                    "Requested Loan",
                    f"₹{loan_amount_rupees:,.0f}"
                )

            with m3:
                st.metric("Model", str(model_name))

            if confidence is not None:
                st.progress(
                    min(max(confidence / 100.0, 0.0), 1.0)
                )

                st.caption(
                    "Model confidence for its predicted class: "
                    f"{confidence:.1f}%"
                )

            # FULL APPLICATION SUMMARY
            result_record = {
                "Full Name": full_name.strip(),
                "Age": age,
                "Gender": gender,
                "Email": email,
                "Phone": phone,
                "Marital Status": marital_status,
                "Residential Status": residential_status,
                "Residential Address": address,
                "Dependents": dependents,
                "Education": education,
                "Employment Type": employment_type,
                "Employer / Business": employer_name,
                "Occupation": occupation,
                "Work Experience (years)": work_experience,
                "Current Employer Years": current_job_years,
                "Self-Employed": self_employed,
                "Applicant Income (₹)": applicant_income,
                "Co-Applicant Income (₹)": coapplicant_income,
                "Existing Monthly EMI (₹)": existing_emi,
                "Other Monthly Obligations (₹)": other_obligations,
                "Savings (₹)": savings,
                "Other Income (₹)": other_income,
                "Credit History": credit_history,
                "Credit Score": credit_score,
                "Existing Loans": existing_loans,
                "Repayment History": repayment_history,
                "Bank Account": bank_account,
                "Banking Relationship (years)": bank_relationship_years,
                "Loan Purpose": loan_purpose,
                "Requested Loan (₹)": loan_amount_rupees,
                "Loan Term (months)": loan_term_months,
                "Property Area": property_area,
                "Property Value (₹)": property_value,
                "Down Payment (₹)": down_payment,
                "City": city,
                "State": state,
                "Postal Code": postal_code,
                "Prediction": decision,
                "Model": str(model_name),
                "Model Confidence (%)": (
                    round(confidence, 2)
                    if confidence is not None
                    else None
                ),
            }

            st.download_button(
                "Download Application Summary (CSV)",
                data=pd.DataFrame([result_record]).to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="loan_application_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.caption(
                "Important: this is an ML-based estimate for "
                "demonstration and decision-support only, not a real "
                "bank approval or a substitute for lender verification."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

            st.info(
                "Confirm that the model file was trained with the "
                "same feature names and preprocessing as the input "
                "built by this app."
            )


# =========================================================
# 8. FOOTER
# =========================================================
st.markdown(
    '<p class="small-note">'
    'Loan Approval Prediction • Demonstration use only'
    '</p>',
    unsafe_allow_html=True
)
