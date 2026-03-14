import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.write("Interactive dashboard for telecom customer churn analysis.")

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = r"C:\Users\sache\Downloads\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
PREDICTION_RESULTS_PATH = "outputs/prediction_results.csv"

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

if df is None:
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.stop()

# Convert TotalCharges safely
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

if "gender" in df.columns:
    gender_options = ["All"] + sorted(df["gender"].dropna().unique().tolist())
    selected_gender = st.sidebar.selectbox("Select Gender", gender_options)
else:
    selected_gender = "All"

if "Contract" in df.columns:
    contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
    selected_contract = st.sidebar.selectbox("Select Contract Type", contract_options)
else:
    selected_contract = "All"

if "InternetService" in df.columns:
    internet_options = ["All"] + sorted(df["InternetService"].dropna().unique().tolist())
    selected_internet = st.sidebar.selectbox("Select Internet Service", internet_options)
else:
    selected_internet = "All"

filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]

if selected_contract != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]

if selected_internet != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]

# -----------------------------
# Top metrics
# -----------------------------
st.subheader("Overview Metrics")

total_customers = len(filtered_df)

if "Churn" in filtered_df.columns:
    churned_customers = len(filtered_df[filtered_df["Churn"] == "Yes"])
    churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
else:
    churned_customers = 0
    churn_rate = 0

avg_monthly_charge = filtered_df["MonthlyCharges"].mean() if "MonthlyCharges" in filtered_df.columns else 0
avg_tenure = filtered_df["tenure"].mean() if "tenure" in filtered_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers}")
col2.metric("Churned Customers", f"{churned_customers}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Avg Monthly Charge", f"{avg_monthly_charge:.2f}")

col5, _ = st.columns(2)
col5.metric("Avg Tenure", f"{avg_tenure:.2f}")

# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

# -----------------------------
# Churn distribution
# -----------------------------
st.subheader("Churn Distribution")

if "Churn" in filtered_df.columns:
    churn_counts = filtered_df["Churn"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts.plot(kind="bar", ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# Monthly charges histogram
# -----------------------------
if "MonthlyCharges" in filtered_df.columns:
    st.subheader("Monthly Charges Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    filtered_df["MonthlyCharges"].hist(ax=ax, bins=30)
    ax.set_title("Monthly Charges Distribution")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -----------------------------
# Tenure by churn
# -----------------------------
if "tenure" in filtered_df.columns and "Churn" in filtered_df.columns:
    st.subheader("Tenure by Churn")

    fig, ax = plt.subplots(figsize=(7, 4))
    filtered_df.boxplot(column="tenure", by="Churn", ax=ax)
    plt.suptitle("")
    ax.set_title("Tenure by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure")
    st.pyplot(fig)

# -----------------------------
# Monthly charges by churn
# -----------------------------
if "MonthlyCharges" in filtered_df.columns and "Churn" in filtered_df.columns:
    st.subheader("Monthly Charges by Churn")

    fig, ax = plt.subplots(figsize=(7, 4))
    filtered_df.boxplot(column="MonthlyCharges", by="Churn", ax=ax)
    plt.suptitle("")
    ax.set_title("Monthly Charges by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")
    st.pyplot(fig)

# -----------------------------
# Churn by contract type
# -----------------------------
if "Contract" in filtered_df.columns and "Churn" in filtered_df.columns:
    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(filtered_df["Contract"], filtered_df["Churn"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn.plot(kind="bar", ax=ax)
    ax.set_title("Churn Percentage by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

# -----------------------------
# Internet service vs churn
# -----------------------------
if "InternetService" in filtered_df.columns and "Churn" in filtered_df.columns:
    st.subheader("Churn by Internet Service")

    internet_churn = pd.crosstab(filtered_df["InternetService"], filtered_df["Churn"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    internet_churn.plot(kind="bar", ax=ax)
    ax.set_title("Churn Percentage by Internet Service")
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

# -----------------------------
# Model comparison section
# -----------------------------
st.subheader("Model Accuracy Comparison")

model_results = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Accuracy": [0.80, 0.74]
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(model_results["Model"], model_results["Accuracy"])
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
st.pyplot(fig)

st.caption("Update these accuracy values later with your actual script results if they differ.")

# -----------------------------
# Prediction results table
# -----------------------------
st.subheader("Prediction Results")

if os.path.exists(PREDICTION_RESULTS_PATH):
    results_df = pd.read_csv(PREDICTION_RESULTS_PATH)
    st.dataframe(results_df.head(50), use_container_width=True)

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction Results CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )
else:
    st.info("Run churn_project.py first to generate outputs/prediction_results.csv")

# -----------------------------
# Insights section
# -----------------------------
st.subheader("Business Insights")

insights = [
    "Customers with shorter tenure are more likely to churn.",
    "Higher monthly charges often correlate with higher churn.",
    "Month-to-month contract customers usually churn more than long-term contract customers.",
    "Contract type and monthly charges are among the most useful churn indicators."
]

for insight in insights:
    st.write(f"- {insight}")