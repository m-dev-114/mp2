import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Agile Dashboard", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard + Agentic AI")

# -------------------------------------------------
# LOAD DATA (DEPLOY SAFE)
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset")

else:
    try:
        df = pd.read_csv("ai_agile_sample_dataset.csv")
        st.info("Using bundled sample dataset")
    except:
        st.warning("No dataset found. Please upload CSV.")
        st.stop()

df = df.fillna(0)

for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
    if col in df.columns:
        df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(df[col]).astype(int)

st.dataframe(df.head())

# -------------------------------------------------
# AGENTIC AI INSIGHT ENGINE
# -------------------------------------------------
def agentic_insights(df):
    insights = []

    if "Percent_Done" in df.columns:
        avg_progress = df["Percent_Done"].mean()
        if avg_progress < 60:
            insights.append("⚠️ Many tasks behind schedule")
        else:
            insights.append("✅ Team progressing well")

    if "Blocked_Stories" in df.columns:
        if df["Blocked_Stories"].mean() > 2:
            insights.append("⚠️ High blocking rate detected")

    if "Current_Workload_Percent" in df.columns:
        if df["Current_Workload_Percent"].mean() > 110:
            insights.append("⚠️ Team overloaded → consider redistribution")

    if "Consecutive_Overloads" in df.columns:
        if df["Consecutive_Overloads"].max() >= 3:
            insights.append("🔥 Burnout risk rising")

    return insights

with st.sidebar:
    st.header("🤖 Agentic AI Manager")
    for i in agentic_insights(df):
        st.write(i)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tabs = st.tabs([
    "Sprint Forecast",
    "Workload Forecast",
    "Resolution Time",
    "Burnout Risk",
    "Resource Allocation"
])

# -------------------------------------------------
# TAB 1 — SPRINT FORECAST
# -------------------------------------------------
with tabs[0]:
    st.header("Sprint Completion Forecast")

    features = [
        'Planned_Story_Points_Sprint',
        'Completed_Story_Points',
        'Percent_Done',
        'Days_Remaining_Sprint',
        'Historical_Velocity',
        'Blocked_Stories',
        'Scope_Change'
    ]

    X = df[features]
    y = df['Success_Label']

    if len(y.unique()) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        inputs = {}
        for col in features:
            inputs[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                key=f"sprint_{col}"
            )

        if st.button("Predict Sprint Success"):
            pred = model.predict([list(inputs.values())])[0]
            prob = model.predict_proba([list(inputs.values())])[0][1]

            if pred:
                st.success(f"Likely to Complete ({prob:.2f})")
            else:
                st.warning(f"Risk of Spillover ({prob:.2f})")

# -------------------------------------------------
# TAB 2 — WORKLOAD FORECAST
# -------------------------------------------------
with tabs[1]:
    st.header("Workload Projection")

    features = [
        'Planned_Story_Points_Resource',
        'Current_Assigned_SP',
        'Historical_Avg_SP',
        'Remaining_Days_Resource',
        'High_Priority_Tasks_Resource',
        'Current_Workload_Percent'
    ]

    X = df[features]
    y = df['Expected_Overload']

    if len(y.unique()) > 1:
        model = RandomForestClassifier()
        model.fit(X, y)

        inputs = {}
        for col in features:
            inputs[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                key=f"workload_{col}"
            )

        if st.button("Predict Overload"):
            pred = model.predict([list(inputs.values())])[0]
            prob = model.predict_proba([list(inputs.values())])[0][1]

            if pred:
                st.warning(f"Overload Risk ({prob:.2f})")
            else:
                st.success(f"Within Capacity ({prob:.2f})")

# -------------------------------------------------
# TAB 3 — RESOLUTION TIME
# -------------------------------------------------
with tabs[2]:
    st.header("Time to Resolve Estimation")

    le_issue = LabelEncoder()
    le_priority = LabelEncoder()

    df["Issue_enc"] = le_issue.fit_transform(df["Issue_Type"])
    df["Priority_enc"] = le_priority.fit_transform(df["Priority"])

    X = df[['Issue_enc','Priority_enc','Original_Estimate_Hours','Story_Points_Issue']]
    y = df['Resolution_Time_Hours']

    model = LinearRegression()
    model.fit(X, y)

    issue = st.selectbox("Issue Type", le_issue.classes_, key="res_issue")
    priority = st.selectbox("Priority", le_priority.classes_, key="res_priority")
    oe = st.number_input("Original Estimate", 1, 100, 8, key="res_oe")
    sp = st.number_input("Story Points", 1, 20, 5, key="res_sp")

    if st.button("Estimate Time"):
        row = [[
            le_issue.transform([issue])[0],
            le_priority.transform([priority])[0],
            oe, sp
        ]]
        pred = model.predict(row)[0]
        st.info(f"Estimated Time: {pred:.1f} hours")

# -------------------------------------------------
# TAB 4 — BURNOUT RISK
# -------------------------------------------------
with tabs[3]:
    st.header("Burnout Risk Alerts")

    features = [
        'Total_SP_This_Sprint',
        'Historical_Avg_SP_Burnout',
        'High_Priority_Tasks_Burnout',
        'Consecutive_Overloads'
    ]

    X = df[features]
    y = df['Risk_Flag']

    if len(y.unique()) > 1:
        model = RandomForestClassifier()
        model.fit(X, y)

        inputs = {}
        for col in features:
            inputs[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                key=f"burnout_{col}"
            )

        if st.button("Check Burnout"):
            pred = model.predict([list(inputs.values())])[0]
            if pred:
                st.warning("Burnout Risk Detected")
            else:
                st.success("Workload Healthy")

# -------------------------------------------------
# TAB 5 — RESOURCE ALLOCATION
# -------------------------------------------------
with tabs[4]:
    st.header("Resource Allocation Suggestions")

    le_summary = LabelEncoder()
    le_labels = LabelEncoder()

    df["Summary_enc"] = le_summary.fit_transform(df["Summary"].astype(str))
    df["Labels_enc"] = le_labels.fit_transform(df["Labels"].astype(str))

    X = df[['Summary_enc','Labels_enc','Original_Estimate_Resource','Story_Points_Resource']]
    y = df['Assignee_Resource']

    model = RandomForestClassifier()
    model.fit(X, y)

    summary = st.text_input("Summary", "Fix bug")
    label = st.text_input("Label", "Bug")
    oe = st.number_input("Original Estimate", 1, 50, 8, key="resalloc_oe")
    sp = st.number_input("Story Points", 1, 20, 5, key="resalloc_sp")

    try:
        summary_enc = le_summary.transform([summary])[0]
    except:
        summary_enc = 0

    try:
        label_enc = le_labels.transform([label])[0]
    except:
        label_enc = 0

    if st.button("Suggest Assignee"):
        pred = model.predict([[summary_enc,label_enc,oe,sp]])[0]
        st.success(f"Recommended: {pred}")
