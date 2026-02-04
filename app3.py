import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Advanced Agentic AI Agile System", layout="wide")
st.title("🧠 Advanced Agentic AI–Driven Agile Project Management")

# --------------------------------------------------
# ADVANCED AGENTIC AI
# --------------------------------------------------
class AdvancedAgileAgent:
    def __init__(self):
        self.memory = []
        self.rewards = 0
        self.metrics = {
            "decisions": 0,
            "avg_risk": [],
            "reward": 0
        }

    def reason(self, signals):
        """
        signals: dict of {objective: (risk_value, confidence_weight)}
        """
        weighted_sum = 0
        explanations = []

        for obj, (risk, weight) in signals.items():
            weighted_sum += risk * weight
            explanations.append(f"{obj}: risk={risk:.2f}, weight={weight}")

        overall_risk = weighted_sum / len(signals)

        # Decision policy
        if overall_risk > 0.7:
            action = "🚨 Critical: Re-plan sprint, reallocate resources, reduce scope"
        elif overall_risk > 0.4:
            action = "⚠️ Preventive: Adjust workload and priorities"
        else:
            action = "✅ Stable: No intervention required"

        self.metrics["decisions"] += 1
        self.metrics["avg_risk"].append(overall_risk)
        self.memory.append((signals, overall_risk, action))

        return overall_risk, action, explanations

    def feedback(self, reward):
        self.metrics["reward"] += reward

    def report(self):
        return {
            "Total Decisions": self.metrics["decisions"],
            "Average Risk Score": np.mean(self.metrics["avg_risk"]) if self.metrics["avg_risk"] else 0,
            "Cumulative Reward": self.metrics["reward"]
        }

# Persist agent
if "agent" not in st.session_state:
    st.session_state.agent = AdvancedAgileAgent()
agent = st.session_state.agent

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload all_objectives_combined.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file).fillna(0)

    for col in ["Success_Label", "Expected_Overload", "Risk_Flag"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col]).astype(int)

    st.success("Dataset loaded")

    # --------------------------------------------------
    # TABS FOR 5 OBJECTIVES
    # --------------------------------------------------
    tabs = st.tabs([
        "1️⃣ Sprint Completion",
        "2️⃣ Workload Projection",
        "3️⃣ Time to Resolve",
        "4️⃣ Burnout Risk",
        "5️⃣ Resource Allocation",
        "🤖 Agentic AI"
    ])

    # ================= OBJECTIVE 1 ====================
    with tabs[0]:
        X1 = df[['Planned_Story_Points_Sprint','Completed_Story_Points',
                 'Percent_Done','Days_Remaining_Sprint',
                 'Historical_Velocity','Blocked_Stories','Scope_Change']]
        y1 = df['Success_Label']

        model1 = LogisticRegression(max_iter=1000)
        model1.fit(X1, y1)
        sprint_prob = model1.predict_proba(X1)[:,1].mean()

        st.metric("Sprint Success Probability", f"{sprint_prob:.2f}")

    # ================= OBJECTIVE 2 ====================
    with tabs[1]:
        X2 = df[['Planned_Story_Points_Resource','Current_Assigned_SP',
                 'Historical_Avg_SP','Remaining_Days_Resource',
                 'High_Priority_Tasks_Resource','Current_Workload_Percent']]
        y2 = df['Expected_Overload']

        model2 = RandomForestClassifier()
        model2.fit(X2, y2)
        overload_prob = model2.predict_proba(X2)[:,1].mean()

        st.metric("Overload Probability", f"{overload_prob:.2f}")

    # ================= OBJECTIVE 3 ====================
    with tabs[2]:
        X3 = pd.get_dummies(df[['Issue_Type','Priority']], drop_first=False)
        X3 = pd.concat([X3, df[['Original_Estimate_Hours','Story_Points_Issue']]], axis=1)
        y3 = df['Resolution_Time_Hours']

        model3 = LinearRegression()
        model3.fit(X3, y3)
        preds = model3.predict(X3)
        delay_risk = np.mean(preds > y3.mean())

        st.metric("Delay Risk Score", f"{delay_risk:.2f}")

    # ================= OBJECTIVE 4 ====================
    with tabs[3]:
        X4 = df[['Total_SP_This_Sprint','Historical_Avg_SP_Burnout',
                 'High_Priority_Tasks_Burnout','Consecutive_Overloads']]
        y4 = df['Risk_Flag']

        model4 = RandomForestClassifier()
        model4.fit(X4, y4)
        burnout_prob = model4.predict_proba(X4)[:,1].mean()

        st.metric("Burnout Probability", f"{burnout_prob:.2f}")

    # ================= OBJECTIVE 5 ====================
    with tabs[4]:
        le1, le2 = LabelEncoder(), LabelEncoder()
        df['Summary_enc'] = le1.fit_transform(df['Summary'].astype(str))
        df['Labels_enc'] = le2.fit_transform(df['Labels'].astype(str))

        X5 = df[['Summary_enc','Labels_enc',
                 'Original_Estimate_Resource','Story_Points_Resource']]
        y5 = df['Assignee_Resource']

        model5 = RandomForestClassifier()
        model5.fit(X5, y5)
        match_conf = accuracy_score(y5, model5.predict(X5))

        st.metric("Resource Match Confidence", f"{match_conf:.2f}")

    # ================= AGENTIC AI ====================
    with tabs[5]:
        st.header("🤖 Advanced Agentic AI Reasoning")

        signals = {
            "Sprint Completion": (1 - sprint_prob, 0.9),
            "Workload Overload": (overload_prob, 0.85),
            "Resolution Delay": (delay_risk, 0.7),
            "Burnout Risk": (burnout_prob, 0.9),
            "Resource Mismatch": (1 - match_conf, 0.6)
        }

        if st.button("🧠 Run Agent"):
            risk, action, explanation = agent.reason(signals)

            st.subheader("📌 Agent Decision")
            st.success(action)

            st.subheader("🧾 Reasoning Trace")
            for e in explanation:
                st.write("•", e)

            st.metric("Overall Project Risk", f"{risk:.2f}")

        st.subheader("🔁 Feedback")
        fb = st.radio("Agent outcome", ["Positive", "Neutral", "Negative"])
        if st.button("Submit Feedback"):
            agent.feedback(1 if fb == "Positive" else -1 if fb == "Negative" else 0)
            st.success("Feedback recorded")

        st.subheader("📊 Agent Performance")
        report = agent.report()
        for k, v in report.items():
            st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)
