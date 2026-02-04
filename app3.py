import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Advanced Agentic AI Agile System",
    layout="wide"
)

st.title("🧠 Advanced Agentic AI for Agile Project Management")

# --------------------------------------------------
# ADVANCED AGENTIC AI
# --------------------------------------------------
class AdvancedAgileAgent:
    def __init__(self):
        self.memory = []
        self.reward = 0.0
        self.decision_count = 0

    def reason(self, signals):
        """
        signals = {
            'sprint_risk': (value, confidence),
            'overload': (value, confidence),
            'burnout': (value, confidence),
            'delay': (value, confidence)
        }
        """

        weighted_score = 0
        explanations = []

        for k, (val, conf) in signals.items():
            weighted_score += val * conf
            explanations.append(f"{k}: risk={val}, confidence={conf:.2f}")

        # Normalize score
        risk_score = weighted_score / len(signals)

        # Policy selection
        if risk_score > 0.65:
            action = "🚨 Critical Intervention: Re-plan sprint + reallocate resources"
        elif risk_score > 0.4:
            action = "⚠️ Preventive Action: Adjust workload and priorities"
        else:
            action = "✅ No action needed"

        self.decision_count += 1
        self.memory.append({
            "signals": signals,
            "risk_score": risk_score,
            "action": action
        })

        return risk_score, action, explanations

    def update_reward(self, feedback):
        if feedback == "Positive":
            self.reward += 1
        elif feedback == "Negative":
            self.reward -= 1

    def metrics(self):
        return {
            "Decisions Made": self.decision_count,
            "Cumulative Reward": self.reward,
            "Avg Risk Score": np.mean([m["risk_score"] for m in self.memory]) if self.memory else 0
        }

# Persist agent across reruns
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
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col]).astype(int)

    st.success("Dataset loaded successfully")

    # --------------------------------------------------
    # ML MODELS → SIGNAL GENERATION
    # --------------------------------------------------
    st.header("📊 Risk Signal Generation (ML Layer)")

    # Sprint risk
    X1 = df[['Planned_Story_Points_Sprint','Completed_Story_Points',
             'Percent_Done','Days_Remaining_Sprint',
             'Historical_Velocity','Blocked_Stories','Scope_Change']]
    y1 = df['Success_Label']

    Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.2)
    sprint_model = LogisticRegression(max_iter=1000)
    sprint_model.fit(Xtr, ytr)
    sprint_prob = sprint_model.predict_proba(Xte)[:,1].mean()

    # Workload risk
    X2 = df[['Planned_Story_Points_Resource','Current_Assigned_SP',
             'Historical_Avg_SP','Remaining_Days_Resource',
             'High_Priority_Tasks_Resource','Current_Workload_Percent']]
    y2 = df['Expected_Overload']

    Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2)
    workload_model = RandomForestClassifier()
    workload_model.fit(Xtr, ytr)
    workload_prob = workload_model.predict_proba(Xte)[:,1].mean()

    # Burnout risk
    X3 = df[['Total_SP_This_Sprint','Historical_Avg_SP_Burnout',
             'High_Priority_Tasks_Burnout','Consecutive_Overloads']]
    y3 = df['Risk_Flag']

    Xtr, Xte, ytr, yte = train_test_split(X3, y3, test_size=0.2)
    burnout_model = RandomForestClassifier()
    burnout_model.fit(Xtr, ytr)
    burnout_prob = burnout_model.predict_proba(Xte)[:,1].mean()

    st.metric("Sprint Failure Probability", f"{sprint_prob:.2f}")
    st.metric("Overload Probability", f"{workload_prob:.2f}")
    st.metric("Burnout Probability", f"{burnout_prob:.2f}")

    # --------------------------------------------------
    # AGENTIC AI REASONING
    # --------------------------------------------------
    st.header("🤖 Autonomous Agentic Reasoning")

    signals = {
        "sprint_risk": (sprint_prob, 0.9),
        "overload": (workload_prob, 0.85),
        "burnout": (burnout_prob, 0.8),
        "delay": (0.5, 0.6)  # simulated signal
    }

    if st.button("🧠 Run Agent"):
        risk_score, action, explanation = agent.reason(signals)

        st.subheader("📌 Agent Decision")
        st.success(action)

        st.subheader("🧾 Reasoning Trace")
        for e in explanation:
            st.write("•", e)

        st.metric("Overall Risk Score", f"{risk_score:.2f}")

    # --------------------------------------------------
    # FEEDBACK LOOP
    # --------------------------------------------------
    st.header("🔁 Human Feedback (Learning Loop)")
    feedback = st.radio("Was the agent decision useful?", ["Neutral", "Positive", "Negative"])

    if st.button("Submit Feedback"):
        agent.update_reward(feedback)
        st.success("Feedback applied to agent learning")

    # --------------------------------------------------
    # AGENT PERFORMANCE DASHBOARD
    # --------------------------------------------------
    st.header("📈 Agent Performance Metrics")

    metrics = agent.metrics()
    for k, v in metrics.items():
        st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)
