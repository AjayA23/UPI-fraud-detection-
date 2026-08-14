import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "upi_fraud_leakage_controlled.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ==================================================
# HEADER
# ==================================================

st.title("💳 UPI Fraud Detection System")
st.markdown(
    "Enter the UPI transaction details manually and check whether "
    "the transaction is **FRAUD** or **GENUINE**."
)

st.divider()

# ==================================================
# TRANSACTION DETAILS
# ==================================================

st.subheader("💰 Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

with col2:
    receiver_account_age = st.number_input(
        "Receiver Account Age (days)",
        min_value=0,
        value=100
    )

with col3:
    receiver_transaction_history = st.number_input(
        "Receiver Transaction History",
        min_value=0,
        value=10
    )

col1, col2, col3 = st.columns(3)

with col1:
    merchant_category_code = st.selectbox(
        "Merchant Category",
        [
            "food",
            "services",
            "retail",
            "utilities",
            "unknown"
        ]
    )

with col2:
    transaction_amount_vs_sender_history = st.number_input(
        "Amount vs Sender History",
        min_value=0.0,
        value=1.0
    )

with col3:
    request_amount_roundness = st.number_input(
        "Request Amount Roundness",
        min_value=0.0,
        value=0.0
    )

# ==================================================
# AUTHENTICATION DETAILS
# ==================================================

st.subheader("🔐 Authentication Details")

col1, col2, col3 = st.columns(3)

with col1:
    authentication_attempt_count = st.number_input(
        "Authentication Attempt Count",
        min_value=0,
        value=1
    )

with col2:
    authentication_attempts = st.number_input(
        "Authentication Attempts",
        min_value=0,
        value=1
    )

with col3:
    pin_entry_method = st.selectbox(
        "PIN Entry Method",
        ["manual", "pasted"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    pin_entry_speed = st.number_input(
        "PIN Entry Speed",
        min_value=0.0,
        value=1.0
    )

with col2:
    otp_request_device_consistency = st.selectbox(
        "OTP Device Consistency",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    time_pressure_indicators = st.selectbox(
        "Time Pressure Indicator",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ==================================================
# DEVICE / BEHAVIOUR DETAILS
# ==================================================

st.subheader("📱 Device & Behaviour Details")

col1, col2, col3 = st.columns(3)

with col1:
    keyboard_input_speed = st.number_input(
        "Keyboard Input Speed",
        min_value=0.0,
        value=1.0
    )

with col2:
    input_timing_consistency = st.number_input(
        "Input Timing Consistency",
        min_value=0.0,
        value=1.0
    )

with col3:
    background_data_usage = st.number_input(
        "Background Data Usage",
        min_value=0.0,
        value=0.0
    )

col1, col2 = st.columns(2)

with col1:
    handle_transaction_history = st.number_input(
        "UPI Handle Transaction History",
        min_value=0,
        value=10
    )

with col2:
    handle_to_description_consistency = st.selectbox(
        "Handle Description Consistency",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ==================================================
# REQUEST DETAILS
# ==================================================

st.subheader("📨 Request Details")

request_acceptance_rate = st.slider(
    "Request Acceptance Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# ==================================================
# PREDICTION BUTTON
# ==================================================

st.divider()

predict_button = st.button(
    "🔍 CHECK TRANSACTION",
    use_container_width=True
)

# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    input_data = pd.DataFrame([{

        "amount": amount,

        "session_duration": 100,

        "authentication_attempts":
            authentication_attempts,

        "receiver_account_age":
            receiver_account_age,

        "receiver_transaction_history":
            receiver_transaction_history,

        "transaction_amount_vs_sender_history":
            transaction_amount_vs_sender_history,

        "geographic_disparity": 0,

        "transaction_time_of_day": 12,

        "merchant_category_code":
            merchant_category_code,

        "session_source": "app",

        "time_between_link_click_and_transaction": 0,

        "dns_lookup_age": 0,

        "recent_app_installs": "[]",

        "input_timing_consistency":
            input_timing_consistency,

        "app_switching_frequency": 0,

        "keyboard_input_speed":
            keyboard_input_speed,

        "input_pause_patterns": 0,

        "screen_active_time": 300,

        "geographic_location_vs_ip": 0,

        "authentication_attempt_count":
            authentication_attempt_count,

        "time_between_otp_generation_and_input": 0,

        "pin_entry_method":
            pin_entry_method,

        "pin_entry_speed":
            pin_entry_speed,

        "otp_request_frequency": 0,

        "otp_request_device_consistency":
            otp_request_device_consistency,

        "transaction_velocity": 0,

        "failed_transaction_count": 0,

        "authorization_method": "pin",

        "transaction_type": "payment",

        "request_amount_roundness":
            request_amount_roundness,

        "request_frequency": 0,

        "request_acceptance_rate":
            request_acceptance_rate,

        "time_to_respond_to_request": 0,

        "time_pressure_indicators":
            time_pressure_indicators,

        "requester_account_age": 100,

        "relationship_to_requester": "unknown",

        "upi_handle_age": 100,

        "handle_similarity_score": 0,

        "handle_contains_official_terms": 0,

        "handle_transaction_history":
            handle_transaction_history,

        "social_media_presence": "none",

        "handle_to_description_consistency":
            handle_to_description_consistency,

        "background_data_usage":
            background_data_usage

    }])

    # ==================================================
    # CHECK FEATURES
    # ==================================================

    expected_features = (
        model.named_steps["preprocessor"]
        .feature_names_in_
    )

    missing_features = set(
        expected_features
    ) - set(
        input_data.columns
    )

    if missing_features:

        st.error(
            f"Model features missing: {missing_features}"
        )

    else:

        input_data = input_data[
            expected_features
        ]

        # ==================================================
        # PREDICT
        # ==================================================

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        # ==================================================
        # RESULT
        # ==================================================

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction == 1:
                st.error("🚨 FRAUD")
            else:
                st.success("✅ GENUINE")

        with col2:

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        with col3:

            if probability >= 0.70:
                risk = "HIGH"
            elif probability >= 0.40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            st.metric(
                "Risk Level",
                risk
            )

        # ==================================================
        # PROBABILITY BAR
        # ==================================================

        st.subheader("Fraud Probability")

        st.progress(
            float(probability)
        )

        # ==================================================
        # MESSAGE
        # ==================================================

        if prediction == 1:

            st.warning(
                "⚠️ This transaction appears suspicious. "
                "Please verify the receiver before making payment."
            )

        else:

            st.info(
                "✅ This transaction appears legitimate "
                "according to the trained model."
            )

        # ==================================================
        # ENTERED DATA
        # ==================================================

        with st.expander("🔎 View Entered Transaction Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "UPI Fraud Detection System | Machine Learning Project"
)