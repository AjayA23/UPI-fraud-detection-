import pandas as pd
import joblib

# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "upi_fraud_leakage_controlled.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 65)
print("              UPI FRAUD DETECTION SYSTEM")
print("=" * 65)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_binary_input(message):
    while True:
        try:
            value = int(input(message))

            if value in [0, 1]:
                return value

            print("Please enter only 0 or 1.")

        except ValueError:
            print("Please enter a valid number.")


def get_rate_input(message):
    while True:
        try:
            value = float(input(message))

            if 0 <= value <= 1:
                return value

            print("Please enter a value between 0 and 1.")

        except ValueError:
            print("Please enter a valid number.")


# ==================================================
# USER INPUT
# ==================================================

amount = float(
    input("Transaction Amount (₹): ")
)

receiver_account_age = int(
    input("Receiver Account Age (days): ")
)

receiver_transaction_history = int(
    input("Receiver Transaction History: ")
)

merchant_category_code = input(
    "Merchant Category (food/services/retail/utilities/unknown): "
).strip().lower()

# Validate merchant category
valid_categories = [
    "food",
    "services",
    "retail",
    "utilities",
    "unknown"
]

if merchant_category_code not in valid_categories:
    merchant_category_code = "unknown"

time_pressure_indicators = get_binary_input(
    "Time Pressure Indicator (0 = No, 1 = Yes): "
)

handle_to_description_consistency = get_binary_input(
    "Handle Description Consistency (0/1): "
)

keyboard_input_speed = float(
    input("Keyboard Input Speed: ")
)

handle_transaction_history = int(
    input("Handle Transaction History: ")
)

request_amount_roundness = float(
    input("Request Amount Roundness: ")
)

transaction_amount_vs_sender_history = float(
    input("Transaction Amount vs Sender History: ")
)

input_timing_consistency = float(
    input("Input Timing Consistency: ")
)

background_data_usage = float(
    input("Background Data Usage: ")
)

otp_request_device_consistency = get_binary_input(
    "OTP Device Consistency (0/1): "
)

pin_entry_speed = float(
    input("PIN Entry Speed: ")
)

authentication_attempt_count = int(
    input("Authentication Attempt Count: ")
)


# ==================================================
# PIN ENTRY METHOD
# ==================================================

while True:

    pin_entry_method = input(
        "PIN Entry Method (manual/pasted): "
    ).strip().lower()

    if pin_entry_method in ["manual", "pasted"]:
        break

    print("Please enter either 'manual' or 'pasted'.")


authentication_attempts = int(
    input("Authentication Attempts: ")
)


# ==================================================
# REQUEST ACCEPTANCE RATE
# ==================================================

request_acceptance_rate = get_rate_input(
    "Request Acceptance Rate (0-1): "
)


# ==================================================
# DEFAULT VALUES
# ==================================================

input_data = pd.DataFrame([{

    # -------------------------------
    # Transaction Features
    # -------------------------------

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

    # -------------------------------
    # Device / Network
    # -------------------------------

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

    # -------------------------------
    # Authentication
    # -------------------------------

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

    # -------------------------------
    # Transaction Behaviour
    # -------------------------------

    "transaction_velocity": 0,

    "failed_transaction_count": 0,

    "authorization_method": "pin",

    "transaction_type": "payment",

    # -------------------------------
    # Request Features
    # -------------------------------

    "request_amount_roundness":
        request_amount_roundness,

    "request_frequency": 0,

    "request_acceptance_rate":
        request_acceptance_rate,

    "time_to_respond_to_request": 0,

    "time_pressure_indicators":
        time_pressure_indicators,

    # -------------------------------
    # Receiver / UPI Handle
    # -------------------------------

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

    # IMPORTANT
    # Background data usage was missing earlier
    "background_data_usage":
        background_data_usage

}])


# ==================================================
# CHECK MODEL FEATURES
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

    print("\nERROR: Missing Features:")
    print(missing_features)

    raise ValueError(
        "Prediction input is missing required model features."
    )


# ==================================================
# REORDER COLUMNS
# ==================================================

input_data = input_data[
    expected_features
]


# ==================================================
# PREDICTION
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

print("\n" + "=" * 65)
print("                 PREDICTION RESULT")
print("=" * 65)

if prediction == 1:

    print("Prediction        : FRAUD")

    print(
        f"Fraud Probability : {probability * 100:.2f}%"
    )

    print("\nWARNING:")
    print("This transaction appears suspicious.")
    print(
        "Verify the receiver before making payment."
    )

else:

    print("Prediction        : GENUINE")

    print(
        f"Fraud Probability : {probability * 100:.2f}%"
    )

    print("\nTransaction appears legitimate.")

print("=" * 65)