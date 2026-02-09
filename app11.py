import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AI Agile Dashboard", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard")


# ------------------------------------------------------------
# SAFE CSV LOADER
# ------------------------------------------------------------

def load_data(file):
    try:
        df = pd.read_csv(file)
        df = df.fillna(0)
        for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(df[col]).astype(int)
        return df
    except:
        st.error("Invalid CSV format")
        st.stop()


# ------------------------------------------------------------
# CACHED MODELS
# ------------------------------------------------------------

@st.cache_resource
def get_sprint_model(X, y):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X, y)
    return pipe


@st.cache_resource
def get_rf_model(X, y):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model


@st.cache_resource
def get_ttr_model(df):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Issue_Type", "Priority"]),
        ("num", "passthrough", ["Original_Estimate_Hours", "Story_Points_Issue"])
    ])

    model = Pipeline([("prep", pre), ("reg", LinearRegression())])
    X = df[["Issue_Type", "Priority", "Original_Estimate_Hours", "Story_Points_Issue"]]
    y = df["Resolution_Time_Hours"]
    model.fit(X, y)
    return model


@st.cache_resource
def get_resource_model(df):
    pre = ColumnTransformer([
        ("summary", TfidfVectorizer(), "Summary"),
        ("labels", TfidfVectorizer(), "Labels"),
        ("num", "passthrough", ["Original_Estimate_Resource", "Story_Points_Resource"])
    ])

    model = Pipeline([("prep", pre), ("clf", RandomForestClassifier(n_estimators=200))])
    X = df[["Summary", "Labels", "Original_Estimate_Resource", "Story_Points_Resource"]]
    y = df["Assignee_Resource"]
    model.fit(X, y)
    return model


# ------------------------------------------------------------
# AGENTIC AI
# ------------------------------------------------------------

def run_agent(df, sprint_model, workload_model, burnout_model):
    results = {}

    X1 = df[[
        'Planned_Story_Points_Sprint',
        'Completed_Story_Points',
        'Percent_Done',
        'Days_Remaining_Sprint',
        'Historical_Velocity',
        'Blocked_Stories',
        'Scope_Change'
    ]]
    results["Sprint Risk"] = float(1 - sprint_model.predict(X1).mean())

    X2 = df[[
        'Planned_Story_Points_Resource',
        'Current_Assigned_SP',
        'Historical_Avg_SP',
        'Remaining_Days_Resource',
        'High_Priority_Tasks_Resource',
        'Current_Workload_Percent'
    ]]
    results["Overload Risk"] = float(workload_model.predict(X2).mean())

    X4 = df[[
        'Total_SP_This_Sprint',
        'Historical_Avg_SP_Burnout',
        'High_Priority_Tasks_Burnout',
        'Consecutive_Overloads'
    ]]
    results["Burnout Risk"] = float(burnout_model.predict(X4).mean())

    actions = []

    if results["Sprint Risk"] > 0.5:
        actions.append("Reduce sprint scope or increase resources")

    if results["Overload Risk"] > 0.5:
        actions.append("Rebalance team workload")

    if results["Burnout Risk"] > 0.4:
        actions.append("Lower high-priority pressure")

    if not actions:
        actions.append("Project health stable")

    return results, actions


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = load_data(uploaded)
    st.success("Data loaded")
    st.dataframe(df.head())

    tabs = st.tabs([
        "Sprint Forecast",
        "Workload",
        "Resolution Time",
        "Burnout",
        "Resource Allocation",
        "AI Project Manager"
    ])

    with tabs[0]:
        X = df[[
            'Planned_Story_Points_Sprint',
            'Completed_Story_Points',
            'Percent_Done',
            'Days_Remaining_Sprint',
            'Historical_Velocity',
            'Blocked_Stories',
            'Scope_Change'
        ]]
        y = df['Success_Label']
        if len(y.unique()) > 1:
            model = get_sprint_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

    with tabs[1]:
        X = df[[
            'Planned_Story_Points_Resource',
            'Current_Assigned_SP',
            'Historical_Avg_SP',
            'Remaining_Days_Resource',
            'High_Priority_Tasks_Resource',
            'Current_Workload_Percent'
        ]]
        y = df['Expected_Overload']
        if len(y.unique()) > 1:
            model = get_rf_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

    with tabs[2]:
        model = get_ttr_model(df)
        X = df[["Issue_Type", "Priority", "Original_Estimate_Hours", "Story_Points_Issue"]]
        y = df["Resolution_Time_Hours"]
        st.write("MSE:", mean_squared_error(y, model.predict(X)))

    with tabs[3]:
        X = df[[
            'Total_SP_This_Sprint',
            'Historical_Avg_SP_Burnout',
            'High_Priority_Tasks_Burnout',
            'Consecutive_Overloads'
        ]]
        y = df['Risk_Flag']
        if len(y.unique()) > 1:
            model = get_rf_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

    with tabs[4]:
        get_resource_model(df)
        st.success("Resource model ready")

    with tabs[5]:
        st.subheader("Autonomous AI Project Manager")

        sprint_model = get_sprint_model(
            df[[
                'Planned_Story_Points_Sprint',
                'Completed_Story_Points',
                'Percent_Done',
                'Days_Remaining_Sprint',
                'Historical_Velocity',
                'Blocked_Stories',
                'Scope_Change'
            ]],
            df['Success_Label']
        )

        workload_model = get_rf_model(
            df[[
                'Planned_Story_Points_Resource',
                'Current_Assigned_SP',
                'Historical_Avg_SP',
                'Remaining_Days_Resource',
                'High_Priority_Tasks_Resource',
                'Current_Workload_Percent'
            ]],
            df['Expected_Overload']
        )

        burnout_model = get_rf_model(
            df[[
                'Total_SP_This_Sprint',
                'Historical_Avg_SP_Burnout',
                'High_Priority_Tasks_Burnout',
                'Consecutive_Overloads'
            ]],
            df['Risk_Flag']
        )

        if st.button("Run Autonomous Analysis"):
            results, actions = run_agent(df, sprint_model, workload_model, burnout_model)

            st.write("System Health")
            st.json(results)

            st.write("AI Recommendations")
            for a in actions:
                st.write("•", a)
