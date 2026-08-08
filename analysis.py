import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==================================================
# 1. LOAD DATASET
# ==================================================

DATASET = r"C:\Users\ajaya\OneDrive\画像\archive (4)\fraud_dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 65)
print("LEAKAGE-CONTROLLED UPI FRAUD DETECTION")
print("=" * 65)

print("\nOriginal Dataset:", df.shape)


# ==================================================
# 2. FRAUD DISTRIBUTION
# ==================================================

print("\n" + "=" * 65)
print("FRAUD DISTRIBUTION")
print("=" * 65)

print("Genuine Transactions :", (df["is_fraud"] == 0).sum())
print("Fraud Transactions   :", (df["is_fraud"] == 1).sum())

fraud_percentage = (
    df["is_fraud"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nFraud Percentage:")
print(fraud_percentage.round(2))


# ==================================================
# 3. REMOVE IDs / TEXT / DIRECT FRAUD INDICATORS
# ==================================================

drop_columns = [

    # IDs / personal identifiers
    "transaction_id",
    "user_id",
    "merchant_id",
    "device_id",
    "ip_address",

    # Free text / high-cardinality fields
    "description",
    "location",
    "timestamp",
    "url_referrer",
    "request_description",
    "request_description_keywords",
    "permissions_granted",
    "recognized_screen_sharing_apps",

    # Direct fraud indicators
    "unusual_device_flag",
    "unusual_ip_flag",
    "unusual_location_flag",
    "unusual_transaction_amount_flag",
    "handle_verification_status",
    "handle_typo_analysis",
    "handle_registration_pattern",
    "business_name_match"
]

df = df.drop(
    columns=drop_columns,
    errors="ignore"
)

print("\nDataset after leakage-control:", df.shape)


# ==================================================
# 4. TARGET
# ==================================================

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("\nFeatures:", X.shape[1])
print("Target:", y.name)


# ==================================================
# 5. FEATURE TYPES
# ==================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))


# ==================================================
# 6. PREPROCESSING
# ==================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==================================================
# 7. TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 65)
print("TRAIN TEST SPLIT")
print("=" * 65)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ==================================================
# 8. RANDOM FOREST MODEL
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)


# ==================================================
# 9. TRAIN MODEL
# ==================================================

print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)

print("Training Leakage-Controlled Random Forest...")

model.fit(X_train, y_train)

print("Training Completed!")


# ==================================================
# 10. PREDICTION
# ==================================================

y_pred = model.predict(X_test)

# Probability of Fraud
y_probability = model.predict_proba(X_test)[:, 1]


# ==================================================
# 11. MODEL PERFORMANCE
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==================================================
# 12. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm)


# ==================================================
# 13. CLASSIFICATION REPORT
# ==================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0
)

print(report)


# ==================================================
# 14. ROC-AUC
# ==================================================

auc_score = roc_auc_score(
    y_test,
    y_probability
)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

print("\n" + "=" * 65)
print("ROC-AUC ANALYSIS")
print("=" * 65)

print(f"AUC Score: {auc_score:.4f}")


# ==================================================
# 15. ROC CURVE
# ==================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - UPI Fraud Detection"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()


# ==================================================
# 16. CONFUSION MATRIX GRAPH
# ==================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Confusion Matrix - Leakage Controlled Model"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "leakage_controlled_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==================================================
# 17. FEATURE IMPORTANCE
# ==================================================

rf_model = model.named_steps["classifier"]

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 65)
print("TOP 20 FEATURES")
print("=" * 65)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ==================================================
# 18. FEATURE IMPORTANCE GRAPH
# ==================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Importance"
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top 15 Features - Leakage Controlled Model"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    "leakage_controlled_feature_importance.png",
    dpi=300
)

plt.show()


# ==================================================
# 19. SAVE MODEL
# ==================================================

MODEL_PATH = "upi_fraud_leakage_controlled.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 65)
print("MODEL SAVED")
print("=" * 65)

print("Model File:", MODEL_PATH)


# ==================================================
# 20. SAVE RESULTS
# ==================================================

with open(
    "leakage_controlled_results.txt",
    "w"
) as file:

    file.write(
        "LEAKAGE-CONTROLLED UPI FRAUD DETECTION\n"
    )

    file.write("=" * 65 + "\n\n")

    file.write(
        f"Dataset Shape: {df.shape}\n"
    )

    file.write(
        f"Training Samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing Samples: {len(X_test)}\n\n"
    )

    file.write(
        "MODEL PERFORMANCE\n"
    )

    file.write("=" * 65 + "\n")

    file.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision : {precision:.4f}\n"
    )

    file.write(
        f"Recall    : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    file.write(
        f"AUC Score : {auc_score:.4f}\n\n"
    )

    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write("=" * 65 + "\n")

    file.write(
        str(cm)
    )

    file.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    file.write("=" * 65 + "\n")

    file.write(
        report
    )

    file.write(
        "\n\nTOP 20 FEATURES\n"
    )

    file.write("=" * 65 + "\n")

    file.write(
        feature_importance
        .head(20)
        .to_string(index=False)
    )


print("\nResults saved:")
print("leakage_controlled_results.txt")

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)