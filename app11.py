import streamlit as st
import pandas as pd
import numpy as np

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
# DATA LOADER
# ------------------------------------------------------------

def load_data(file):
    df = pd.read_csv(file).fillna(0)
    for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(df[col]).astype(int)
    return df


# ------------------------------------------------------------
# MODEL BUILDERS (CACHED)
# ------------------------------------------------------------

@st.cache_resource
def build_sprint_model(X, y):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


@st.cache_resource
def build_rf_model(X, y):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model


@st.cache_resource
def build_ttr_model(df):
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
def build_resource_model(df):
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
        actions.append("Reduce sprint scope or add resources")
    if results["Overload Risk"] > 0.5:
        actions.append("Rebalance workload")
    if results["Burnout Risk"] > 0.4:
        actions.append("Lower high priority pressure")
    if not actions:
        actions.append("Project health stable")

    return results, actions


# ------------------------------------------------------------
# UI START
# ------------------------------------------------------------

uploaded = st.file_uploader("Upload Combined CSV", type="csv")

if uploaded:
    df = load_data(uploaded)
    st.success("Data Loaded")
    st.dataframe(df.head())

    tabs = st.tabs([
        "Sprint Forecast",
        "Workload",
        "Resolution Time",
        "Burnout",
        "Resource Allocation",
        "AI Project Manager"
    ])

    # ------------------------------------------------------------
    # SPRINT
    # ------------------------------------------------------------
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
            model = build_sprint_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

            st.subheader("Predict Sprint Outcome")

            psp = st.number_input("Planned SP", 1, 200, 40)
            csp = st.number_input("Completed SP", 0, 200, 30)
            pdone = st.slider("Percent Done", 0.0, 100.0, 70.0)
            dr = st.number_input("Days Remaining", 0, 30, 5)
            hv = st.number_input("Velocity", 1, 100, 35)
            bs = st.number_input("Blocked Stories", 0, 10, 1)
            sc = st.number_input("Scope Change", -20, 20, 0)

            if st.button("Predict Sprint Completion"):
                row = pd.DataFrame([{
                    'Planned_Story_Points_Sprint': psp,
                    'Completed_Story_Points': csp,
                    'Percent_Done': pdone,
                    'Days_Remaining_Sprint': dr,
                    'Historical_Velocity': hv,
                    'Blocked_Stories': bs,
                    'Scope_Change': sc
                }])
                pred = model.predict(row)[0]
                prob = model.predict_proba(row)[0][1]
                st.success(f"Probability: {prob:.2f}")

    # ------------------------------------------------------------
    # WORKLOAD
    # ------------------------------------------------------------
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
            model = build_rf_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

            st.subheader("Predict Overload")

            psp = st.number_input("Planned SP", 1, 100, 30)
            casp = st.number_input("Assigned SP", 0, 100, 40)
            hasp = st.number_input("Historical Avg", 1, 100, 30)
            rdr = st.number_input("Remaining Days", 1, 30, 5)
            hpt = st.number_input("High Priority Tasks", 0, 10, 2)
            cwp = st.number_input("Workload %", 0, 200, 120)

            if st.button("Predict Overload"):
                row = pd.DataFrame([{
                    'Planned_Story_Points_Resource': psp,
                    'Current_Assigned_SP': casp,
                    'Historical_Avg_SP': hasp,
                    'Remaining_Days_Resource': rdr,
                    'High_Priority_Tasks_Resource': hpt,
                    'Current_Workload_Percent': cwp
                }])
                pred = model.predict(row)[0]
                st.write("Overload Risk:", pred)

    # ------------------------------------------------------------
    # RESOLUTION TIME
    # ------------------------------------------------------------
    with tabs[2]:
        model = build_ttr_model(df)
        X = df[["Issue_Type", "Priority", "Original_Estimate_Hours", "Story_Points_Issue"]]
        y = df["Resolution_Time_Hours"]
        st.write("MSE:", mean_squared_error(y, model.predict(X)))

        issue = st.selectbox("Issue Type", ["Bug", "Story", "Task"])
        pri = st.selectbox("Priority", ["Low", "Medium", "High"])
        oe = st.number_input("Original Estimate", 1, 100, 8)
        sp = st.number_input("Story Points", 1, 20, 5)

        if st.button("Estimate Time"):
            row = pd.DataFrame([{
                "Issue_Type": issue,
                "Priority": pri,
                "Original_Estimate_Hours": oe,
                "Story_Points_Issue": sp
            }])
            pred = model.predict(row)[0]
            st.info(f"{pred:.1f} hours")

    # ------------------------------------------------------------
    # BURNOUT
    # ------------------------------------------------------------
    with tabs[3]:
        X = df[[
            'Total_SP_This_Sprint',
            'Historical_Avg_SP_Burnout',
            'High_Priority_Tasks_Burnout',
            'Consecutive_Overloads'
        ]]
        y = df['Risk_Flag']

        if len(y.unique()) > 1:
            model = build_rf_model(X, y)
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))

            tsp = st.number_input("Total SP", 0, 100, 40)
            hasp = st.number_input("Historical Avg", 1, 100, 25)
            hpt = st.number_input("High Priority Tasks", 0, 10, 2)
            co = st.number_input("Consecutive Overloads", 0, 5, 2)

            if st.button("Check Burnout"):
                row = pd.DataFrame([[tsp, hasp, hpt, co]],
                                   columns=X.columns)
                st.write("Risk:", model.predict(row)[0])

    # ------------------------------------------------------------
    # RESOURCE ALLOCATION
    # ------------------------------------------------------------
    with tabs[4]:
        model = build_resource_model(df)

        summary = st.text_input("Summary", "Fix login bug")
        label = st.text_input("Label", "Bug")
        oe = st.number_input("Estimate", 1, 50, 8)
        sp = st.number_input("Story Points", 1, 20, 5)

        if st.button("Suggest Assignee"):
            row = pd.DataFrame([{
                "Summary": summary,
                "Labels": label,
                "Original_Estimate_Resource": oe,
                "Story_Points_Resource": sp
            }])
            st.success(model.predict(row)[0])

    # ------------------------------------------------------------
    # AGENTIC AI
    # ------------------------------------------------------------
    with tabs[5]:
        st.subheader("Autonomous AI Project Manager")

        sprint_model = build_sprint_model(
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

        workload_model = build_rf_model(
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

        burnout_model = build_rf_model(
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
            st.json(results)
            for a in actions:
                st.write("•", a)
