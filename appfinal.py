import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Agile Dashboard", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard + Agentic AI")

# -------------------------------------------------
# LOAD DATA
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

# Convert Yes/No labels
for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
    if col in df.columns:
        df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(df[col]).astype(int)

st.dataframe(df.head())

# -------------------------------------------------
# AGENTIC AI INSIGHTS
# -------------------------------------------------
def agentic_insights(df):
    insights = []

    if "Percent_Done" in df.columns:
        if df["Percent_Done"].mean() < 60:
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
    for insight in agentic_insights(df):
        st.write(insight)

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

# =================================================
# TAB 1 — SPRINT FORECAST
# =================================================
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

    if all(col in df.columns for col in features + ['Success_Label']):
        X = df[features]
        y = df['Success_Label']

        if len(y.unique()) > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            @st.cache_resource
            def train_sprint_model():
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                return model

            model = train_sprint_model()

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.caption(f"Model Accuracy: {acc:.2f}")

            inputs = []
            for col in features:
                val = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max())
                )
                inputs.append(val)

            if st.button("Predict Sprint Success"):
                inputs_scaled = scaler.transform([inputs])
                pred = model.predict(inputs_scaled)[0]
                prob = model.predict_proba(inputs_scaled)[0][1]

                if pred:
                    st.success(f"Likely to Complete ({prob:.2f})")
                else:
                    st.warning(f"Risk of Spillover ({prob:.2f})")
        else:
            st.warning("Not enough label variation for training.")
    else:
        st.warning("Required columns missing.")

# =================================================
# TAB 2 — WORKLOAD FORECAST
# =================================================
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

    if all(col in df.columns for col in features + ['Expected_Overload']):
        X = df[features]
        y = df['Expected_Overload']

        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            @st.cache_resource
            def train_workload_model():
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                return model

            model = train_workload_model()

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.caption(f"Model Accuracy: {acc:.2f}")

            inputs = []
            for col in features:
                val = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max()),
                    key=f"work_{col}"
                )
                inputs.append(val)

            if st.button("Predict Overload"):
                pred = model.predict([inputs])[0]
                prob = model.predict_proba([inputs])[0][1]

                if pred:
                    st.warning(f"Overload Risk ({prob:.2f})")
                else:
                    st.success(f"Within Capacity ({prob:.2f})")
        else:
            st.warning("Not enough label variation.")
    else:
        st.warning("Required columns missing.")

# =================================================
# TAB 3 — RESOLUTION TIME
# =================================================
with tabs[2]:
    st.header("Time to Resolve Estimation")

    if all(col in df.columns for col in
           ["Issue_Type","Priority","Original_Estimate_Hours",
            "Story_Points_Issue","Resolution_Time_Hours"]):

        le_issue = LabelEncoder()
        le_priority = LabelEncoder()

        df["Issue_enc"] = le_issue.fit_transform(df["Issue_Type"])
        df["Priority_enc"] = le_priority.fit_transform(df["Priority"])

        X = df[['Issue_enc','Priority_enc',
                'Original_Estimate_Hours','Story_Points_Issue']]
        y = df['Resolution_Time_Hours']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.caption(f"Model MSE: {mse:.2f}")

        issue = st.selectbox("Issue Type", le_issue.classes_)
        priority = st.selectbox("Priority", le_priority.classes_)
        oe = st.number_input("Original Estimate", 1, 100, 8)
        sp = st.number_input("Story Points", 1, 20, 5)

        if st.button("Estimate Time"):
            row = [[
                le_issue.transform([issue])[0],
                le_priority.transform([priority])[0],
                oe, sp
            ]]
            pred = model.predict(row)[0]
            st.info(f"Estimated Time: {pred:.1f} hours")
    else:
        st.warning("Required columns missing.")

# =================================================
# TAB 4 — BURNOUT RISK
# =================================================
with tabs[3]:
    st.header("Burnout Risk Alerts")

    features = [
        'Total_SP_This_Sprint',
        'Historical_Avg_SP_Burnout',
        'High_Priority_Tasks_Burnout',
        'Consecutive_Overloads'
    ]

    if all(col in df.columns for col in features + ['Risk_Flag']):
        X = df[features]
        y = df['Risk_Flag']

        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.caption(f"Model Accuracy: {acc:.2f}")

            inputs = []
            for col in features:
                val = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max()),
                    key=f"burn_{col}"
                )
                inputs.append(val)

            if st.button("Check Burnout"):
                pred = model.predict([inputs])[0]
                if pred:
                    st.warning("Burnout Risk Detected")
                else:
                    st.success("Workload Healthy")
        else:
            st.warning("Not enough label variation.")
    else:
        st.warning("Required columns missing.")

# =================================================
# TAB 5 — RESOURCE ALLOCATION
# =================================================
with tabs[4]:
    st.header("Resource Allocation Suggestions")

    if all(col in df.columns for col in
           ["Summary","Labels",
            "Original_Estimate_Resource",
            "Story_Points_Resource",
            "Assignee_Resource"]):

        le_summary = LabelEncoder()
        le_labels = LabelEncoder()

        df["Summary_enc"] = le_summary.fit_transform(df["Summary"].astype(str))
        df["Labels_enc"] = le_labels.fit_transform(df["Labels"].astype(str))

        X = df[['Summary_enc','Labels_enc',
                'Original_Estimate_Resource','Story_Points_Resource']]
        y = df['Assignee_Resource']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.caption(f"Model Accuracy: {acc:.2f}")

        summary = st.text_input("Summary", "Fix bug")
        label = st.text_input("Label", "Bug")
        oe = st.number_input("Original Estimate", 1, 50, 8)
        sp = st.number_input("Story Points", 1, 20, 5)

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
    else:
        st.warning("Required columns missing.")
