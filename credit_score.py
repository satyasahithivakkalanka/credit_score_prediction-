# importing libraries
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)

# helpers 

def detect_target(df):
    candidates = [
        "Credit_Score", "credit_score",
        "Risk_Flag", "risk_flag",
        "Default", "default",
        "Loan_Status", "loan_status",
        "Target", "target", "label"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Target column not found. Rename label to one of: " + ", ".join(candidates))

def detect_id(df):
    for c in ["ID", "Id", "id", "Customer_ID", "CustomerId", "customer_id", "case_id"]:
        if c in df.columns:
            return c
    return None

def coerce_numeric(series):
    # keep digits, dot, minus; everything else becomes empty then NaN
    cleaned = series.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def parse_credit_history_to_months(series):
    yrs = pd.to_numeric(series.astype(str).str.extract(r"(\d+)\s*Years?", expand=False), errors="coerce").fillna(0)
    mos = pd.to_numeric(series.astype(str).str.extract(r"(\d+)\s*Months?", expand=False), errors="coerce").fillna(0)
    return (yrs * 12 + mos).astype(float)

def engineer_type_of_loan(df):
    # Replaces the massive-cardi column with compact, useful features
    if "Type_of_Loan" not in df.columns:
        return df
    out = df.copy()
    s = out["Type_of_Loan"].fillna("")
    parts = s.astype(str).str.split(",", regex=False)

    out["num_loan_types"] = parts.apply(lambda lst: len([p.strip() for p in lst if p.strip()]))

    common = [
        "Auto Loan","Credit-Builder Loan","Personal Loan","Student Loan",
        "Home Equity Loan","Debt Consolidation Loan","Payday Loan","Mortgage","Credit Card Loan"
    ]
    def norm(name):
        return re.sub(r"[^A-Za-z]+", "_", name).strip("_").lower()

    for name in common:
        col = "has_" + norm(name)
        out[col] = parts.apply(lambda lst: int(any(name == p.strip() for p in lst)))

    out = out.drop(columns=["Type_of_Loan"])
    return out

def clean_frame(df):
    df = df.copy()

    # drop hard identifiers
    for c in ["ID", "Customer_ID", "SSN", "Name"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # coerce numeric-like object columns
    numeric_like = [
        "Age","Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts",
        "Num_Credit_Card","Interest_Rate","Num_of_Loan","Delay_from_due_date",
        "Num_of_Delayed_Payment","Changed_Credit_Limit","Num_Credit_Inquiries",
        "Credit_Utilization_Ratio","Total_EMI_per_month","Outstanding_Debt",
        "Amount_invested_monthly","Monthly_Balance"
    ]
    for c in numeric_like:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # age sanity
    if "Age" in df.columns:
        df.loc[(df["Age"] < 10) | (df["Age"] > 100), "Age"] = np.nan

    # parse credit history
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = parse_credit_history_to_months(df["Credit_History_Age"])
        df.drop(columns=["Credit_History_Age"], inplace=True)

    # clean placeholder underscores in a few categoricals
    for c in ["Credit_Mix", "Payment_of_Min_Amount"]:
        if c in df.columns:
            df[c] = df[c].replace({"_": np.nan})

    # compact replacement for Type_of_Loan
    df = engineer_type_of_loan(df)

    return df

# paths 

train_path = "train.csv" if os.path.exists("train.csv") else "/mnt/data/train.csv"
test_path  = "test.csv"  if os.path.exists("test.csv")  else "/mnt/data/test.csv"

# load 

train_df = pd.read_csv(train_path, low_memory=False)
test_df  = pd.read_csv(test_path,  low_memory=False)

print("Previewing train data")
print(train_df.head())

target_col = detect_target(train_df)
id_col = detect_id(test_df)
print(f"\nDetected target column: {target_col}")
print(f"Detected id column for submission: {id_col if id_col else '[using index]'}")

# clean both train and test

train_c = clean_frame(train_df)
test_c  = clean_frame(test_df)

X_full = train_c.drop(columns=[target_col])
y_full = train_df[target_col]  # target from original (unchanged)

# infer dtypes after cleaning
num_cols = X_full.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_full.select_dtypes(include=["object","category","bool"]).columns.tolist()

print("\nListing numeric features:", num_cols)
print("Listing categorical features:", cat_cols)

# split 

X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# preprocessors 

numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))  # safe with sparse output downstream
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# models

# fast, strong baseline for mixed data without giant one-hot
hgb = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=None,
    max_iter=200,
    random_state=42
)

hgb_pipe = Pipeline(steps=[("prep", preprocess), ("model", hgb)])

# optional light tree as a quick sanity check (very fast)
dt = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=20
)
dt_pipe = Pipeline(steps=[("prep", preprocess), ("model", dt)])

# train 

print("\nTraining HistGradientBoosting")
hgb_pipe.fit(X_train, y_train)
hgb_pred = hgb_pipe.predict(X_valid)

print("Training DecisionTree (sanity baseline)")
dt_pipe.fit(X_train, y_train)
dt_pred = dt_pipe.predict(X_valid)

# evaluate 

def evaluate(y_true, y_pred, title):
    print(f"\n{title}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    try:
        print("Macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    except Exception:
        pass
    print("\nClassification report")
    print(classification_report(y_true, y_pred))
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(values_format="d")
        plt.title(title + " - Confusion Matrix")
        plt.show()
    except Exception as e:
        print(f"[Plot skipped: {e}]")

evaluate(y_valid, hgb_pred, "HistGradientBoosting - Validation Performance")
evaluate(y_valid, dt_pred,  "DecisionTree - Validation Performance")

# pick best by macro F1 

hgb_f1 = f1_score(y_valid, hgb_pred, average="macro")
dt_f1  = f1_score(y_valid, dt_pred,  average="macro")
best_pipe = hgb_pipe if hgb_f1 >= dt_f1 else dt_pipe
best_name = "HistGradientBoosting" if best_pipe is hgb_pipe else "DecisionTree"
print(f"\nSelected best model: {best_name}")

# refit on full training 

print("\nRefitting best model on full training data")
best_pipe.fit(X_full, y_full)

# predict test

print("\nGenerating predictions for test.csv")
test_pred = best_pipe.predict(test_c.drop(columns=[c for c in [] if c in test_c.columns], errors="ignore"))

# submission

print("\nPreparing submission file")
if id_col and id_col in test_df.columns:
    sub = pd.DataFrame({id_col: test_df[id_col], target_col: test_pred})
else:
    sub = pd.DataFrame({"row_id": np.arange(len(test_df)), target_col: test_pred})

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path}")

print("\nPrediction distribution on test set")
print(sub[target_col].value_counts())
