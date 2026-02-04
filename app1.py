import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="AI Agile Dashboard with Agentic AI", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard (Agentic AI Enabled)")

# -------------------------------------------------------
# AGENTIC AI CLASS
# -------------------------------------------------------
class AgileAIAgent:
    def __init__(self):
        self.memory = []

    def observe(self, context):
        self.context = context

    def decide(self):
        actions = []

        if self.context.get("sprint_risk"):
            actions.append("Reduce sprint scope or add additional resources")

        if self.context.get("overload"):
            actions.append("Reassign tasks from overloaded resources")

        if self.context.get("burnout"):
            actions.append("Recommend cooldown period or workload redistribution")

        if self.context.get("delay_risk"):
            actions.append("Escalate issue priority and allocate senior developer")

        if not actions:
            actions.append("System operating optimally — no action required")

        self.memory.append((self.context, actions))
        return actions

    def explain(self):
        return f"Agent reasoning based on observed signals: {self.context}"

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload the Combined CSV for All Objectives", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.fillna(0)

    for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(df[col]).astype(int)

    st.success("✅ File uploaded successfully!")
    st.write(df.head())

    # -------------------------------------------------------
    # TABS
    # -------------------------------------------------------
    tabs = st.tabs([
        "1️⃣ Sprint Completion Forecast",
        "2️⃣ Workload Projection Forecast",
        "3️⃣ Time to Resolve Estimation",
        "4️⃣ Burnout Risk Alerts",
        "5️⃣ Resource Allocation Suggestions",
        "🤖 Agentic AI Decision Engine"
    ])

    # =======================================================
    # OBJECTIVE 1: SPRINT COMPLETION
    # =======================================================
    with tabs[0]:
        st.header("📌 Sprint Completion Forecast")

        try:
            X1 = df[['Planned_Story_Points_Sprint','Completed_Story_Points',
                     'Percent_Done','Days_Remaining_Sprint',
                     'Historical_Velocity','Blocked_Stories','Scope_Change']]
            y1 = df['Success_Label']

            if len(y1.unique()) > 1:
                X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
                sprint_model = LogisticRegression(max_iter=1000)
                sprint_model.fit(X1_train, y1_train)

                y_pred = sprint_model.predict(X1_test)
                st.write("Accuracy:", accuracy_score(y1_test, y_pred))
                st.text(classification_report(y1_test, y_pred))
        except Exception as e:
            st.error(e)

    # =======================================================
    # OBJECTIVE 2: WORKLOAD PROJECTION
    # =======================================================
    with tabs[1]:
        st.header("📌 Workload Projection Forecast")

        try:
            X2 = df[['Planned_Story_Points_Resource','Current_Assigned_SP',
                     'Historical_Avg_SP','Remaining_Days_Resource',
                     'High_Priority_Tasks_Resource','Current_Workload_Percent']]
            y2 = df['Expected_Overload']

            if len(y2.unique()) > 1:
                X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)
                workload_model = RandomForestClassifier()
                workload_model.fit(X2_train, y2_train)

                y_pred = workload_model.predict(X2_test)
                st.write("Accuracy:", accuracy_score(y2_test, y_pred))
                st.text(classification_report(y2_test, y_pred))
        except Exception as e:
            st.error(e)

    # =======================================================
    # OBJECTIVE 3: TIME TO RESOLVE
    # =======================================================
    with tabs[2]:
        st.header("📌 Time to Resolve Estimation")

        try:
            X3 = pd.get_dummies(df[['Issue_Type','Priority']], drop_first=False)
            X3 = pd.concat([X3, df[['Original_Estimate_Hours','Story_Points_Issue']]], axis=1)
            y3 = df['Resolution_Time_Hours']

            X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2)
            ttr_model = LinearRegression()
            ttr_model.fit(X3_train, y3_train)

            preds = ttr_model.predict(X3_test)
            st.write("MSE:", mean_squared_error(y3_test, preds))
        except Exception as e:
            st.error(e)

    # =======================================================
    # OBJECTIVE 4: BURNOUT RISK
    # =======================================================
    with tabs[3]:
        st.header("📌 Burnout Risk Alerts")

        try:
            X4 = df[['Total_SP_This_Sprint','Historical_Avg_SP_Burnout',
                     'High_Priority_Tasks_Burnout','Consecutive_Overloads']]
            y4 = df['Risk_Flag']

            if len(y4.unique()) > 1:
                X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2)
                burnout_model = RandomForestClassifier()
                burnout_model.fit(X4_train, y4_train)

                y_pred = burnout_model.predict(X4_test)
                st.write("Accuracy:", accuracy_score(y4_test, y_pred))
                st.text(classification_report(y4_test, y_pred))
        except Exception as e:
            st.error(e)

    # =======================================================
    # OBJECTIVE 5: RESOURCE ALLOCATION
    # =======================================================
    with tabs[4]:
        st.header("📌 Resource Allocation Suggestions")

        try:
            le1 = LabelEncoder()
            le2 = LabelEncoder()

            df['Summary_enc'] = le1.fit_transform(df['Summary'].astype(str))
            df['Labels_enc'] = le2.fit_transform(df['Labels'].astype(str))

            X5 = df[['Summary_enc','Labels_enc','Original_Estimate_Resource','Story_Points_Resource']]
            y5 = df['Assignee_Resource']

            X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2)
            alloc_model = RandomForestClassifier()
            alloc_model.fit(X5_train, y5_train)

            y_pred = alloc_model.predict(X5_test)
            st.write("Accuracy:", accuracy_score(y5_test, y_pred))
        except Exception as e:
            st.error(e)

    # =======================================================
    # AGENTIC AI TAB
    # =======================================================
    with tabs[5]:
        st.header("🤖 Agentic AI – Autonomous Decision Engine")

        agent = AgileAIAgent()

        sprint_risk = st.checkbox("Sprint Failure Risk")
        overload = st.checkbox("Resource Overload")
        burnout = st.checkbox("Burnout Risk")
        delay_risk = st.checkbox("Delay in Issue Resolution")

        context = {
            "sprint_risk": sprint_risk,
            "overload": overload,
            "burnout": burnout,
            "delay_risk": delay_risk
        }

        agent.observe(context)

        if st.button("🧠 Run Agent Reasoning"):
            actions = agent.decide()

            st.subheader("✅ Agent Decisions")
            for a in actions:
                st.success(a)

            st.subheader("📝 Explanation")
            st.code(agent.explain())
