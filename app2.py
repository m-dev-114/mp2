import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Agile Dashboard with Agentic AI",
    layout="wide"
)

st.title("🚀 AI Agile Project Management Dashboard (Agentic AI Enabled)")

# --------------------------------------------------
# AGENTIC AI CLASS WITH METRICS
# --------------------------------------------------
class AgileAIAgent:
    def __init__(self):
        self.memory = []
        self.metrics = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "false_alerts": 0,
            "successful_interventions": 0,
            "response_times": []
        }

    def observe(self, context):
        self.context = context
        self.start_time = time.time()

    def decide(self):
        self.metrics["total_decisions"] += 1
        actions = []

        if self.context.get("sprint_risk"):
            actions.append("Reduce sprint scope or add additional resources")

        if self.context.get("overload"):
            actions.append("Reassign tasks from overloaded resources")

        if self.context.get("burnout"):
            actions.append("Recommend cooldown period or workload redistribution")

        if self.context.get("delay_risk"):
            actions.append("Escalate issue priority and assign senior developer")

        if not actions:
            actions.append("No action required — system stable")

        response_time = time.time() - self.start_time
        self.metrics["response_times"].append(response_time)

        self.memory.append((self.context, actions))
        return actions

    def feedback(self, correct, effective):
        if correct:
            self.metrics["correct_decisions"] += 1
        else:
            self.metrics["false_alerts"] += 1

        if effective:
            self.metrics["successful_interventions"] += 1

    def performance_report(self):
        if self.metrics["total_decisions"] == 0:
            return None

        return {
            "Decision Accuracy":
                self.metrics["correct_decisions"] / self.metrics["total_decisions"],
            "False Alert Rate":
                self.metrics["false_alerts"] / self.metrics["total_decisions"],
            "Intervention Effectiveness":
                self.metrics["successful_interventions"] / self.metrics["total_decisions"],
            "Avg Response Time (sec)":
                np.mean(self.metrics["response_times"])
        }

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload the Combined CSV for All Objectives",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.fillna(0)

    for col in ["Success_Label", "Expected_Overload", "Risk_Flag"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col]).astype(int)

    st.success("✅ File uploaded successfully")
    st.write(df.head())

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tabs = st.tabs([
        "1️⃣ Sprint Completion",
        "2️⃣ Workload Projection",
        "3️⃣ Time to Resolve",
        "4️⃣ Burnout Risk",
        "5️⃣ Resource Allocation",
        "🤖 Agentic AI Engine"
    ])

    # --------------------------------------------------
    # OBJECTIVE 1
    # --------------------------------------------------
    with tabs[0]:
        st.header("📌 Sprint Completion Forecast")
        X = df[['Planned_Story_Points_Sprint','Completed_Story_Points',
                'Percent_Done','Days_Remaining_Sprint',
                'Historical_Velocity','Blocked_Stories','Scope_Change']]
        y = df['Success_Label']

        if y.nunique() > 1:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
            model = LogisticRegression(max_iter=1000)
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            st.write("Accuracy:", accuracy_score(yte, preds))
            st.text(classification_report(yte, preds))

    # --------------------------------------------------
    # OBJECTIVE 2
    # --------------------------------------------------
    with tabs[1]:
        st.header("📌 Workload Projection Forecast")
        X = df[['Planned_Story_Points_Resource','Current_Assigned_SP',
                'Historical_Avg_SP','Remaining_Days_Resource',
                'High_Priority_Tasks_Resource','Current_Workload_Percent']]
        y = df['Expected_Overload']

        if y.nunique() > 1:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier()
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            st.write("Accuracy:", accuracy_score(yte, preds))

    # --------------------------------------------------
    # OBJECTIVE 3
    # --------------------------------------------------
    with tabs[2]:
        st.header("📌 Time to Resolve Estimation")
        X = pd.get_dummies(df[['Issue_Type','Priority']], drop_first=False)
        X = pd.concat([X, df[['Original_Estimate_Hours','Story_Points_Issue']]], axis=1)
        y = df['Resolution_Time_Hours']

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        st.write("MSE:", mean_squared_error(yte, preds))

    # --------------------------------------------------
    # OBJECTIVE 4
    # --------------------------------------------------
    with tabs[3]:
        st.header("📌 Burnout Risk Alerts")
        X = df[['Total_SP_This_Sprint','Historical_Avg_SP_Burnout',
                'High_Priority_Tasks_Burnout','Consecutive_Overloads']]
        y = df['Risk_Flag']

        if y.nunique() > 1:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier()
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            st.write("Accuracy:", accuracy_score(yte, preds))

    # --------------------------------------------------
    # OBJECTIVE 5
    # --------------------------------------------------
    with tabs[4]:
        st.header("📌 Resource Allocation Suggestions")
        le1, le2 = LabelEncoder(), LabelEncoder()

        df['Summary_enc'] = le1.fit_transform(df['Summary'].astype(str))
        df['Labels_enc'] = le2.fit_transform(df['Labels'].astype(str))

        X = df[['Summary_enc','Labels_enc',
                'Original_Estimate_Resource','Story_Points_Resource']]
        y = df['Assignee_Resource']

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        st.write("Accuracy:", accuracy_score(yte, preds))

    # --------------------------------------------------
    # AGENTIC AI TAB
    # --------------------------------------------------
    with tabs[5]:
        st.header("🤖 Agentic AI – Autonomous Decision Engine")

        agent = AgileAIAgent()

        sprint_risk = st.checkbox("Sprint Failure Risk")
        overload = st.checkbox("Resource Overload")
        burnout = st.checkbox("Burnout Risk")
        delay = st.checkbox("Issue Resolution Delay")

        context = {
            "sprint_risk": sprint_risk,
            "overload": overload,
            "burnout": burnout,
            "delay_risk": delay
        }

        agent.observe(context)

        if st.button("🧠 Run Agent Reasoning"):
            actions = agent.decide()
            st.subheader("✅ Agent Decisions")
            for a in actions:
                st.success(a)

        st.subheader("📝 Feedback")
        correct = st.radio("Was the decision correct?", ["Yes","No"])
        effective = st.radio("Did it reduce risk?", ["Yes","No"])

        if st.button("Submit Feedback"):
            agent.feedback(correct=="Yes", effective=="Yes")
            st.success("Feedback recorded")

        st.subheader("📊 Agent Performance Metrics")
        report = agent.performance_report()
        if report:
            for k,v in report.items():
                st.metric(k, f"{v:.2f}")
        else:
            st.info("No agent decisions recorded yet")
