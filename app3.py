import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Agile Dashboard", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard")

# --------------------------------------------------
# AGENTIC AI SUPERVISOR
# --------------------------------------------------
class AgileAgent:
    def __init__(self):
        self.memory = []
        self.metrics = {"decisions": 0}

    def decide(self, signals):
        weights = {
            "sprint": 0.9,
            "overload": 0.85,
            "delay": 0.7,
            "burnout": 0.9,
            "resource": 0.6
        }

        weighted_risk = sum(signals[k] * weights[k] for k in signals) / len(signals)

        if weighted_risk > 0.7:
            action = "🚨 Critical: Re-plan sprint, reduce scope, reallocate resources"
        elif weighted_risk > 0.4:
            action = "⚠️ Preventive: Balance workload and adjust priorities"
        else:
            action = "✅ Stable: No action required"

        self.memory.append({
            "signals": signals,
            "risk": weighted_risk,
            "action": action
        })

        self.metrics["decisions"] += 1
        return weighted_risk, action

# Persist agent
if "agent" not in st.session_state:
    st.session_state.agent = AgileAgent()

agent = st.session_state.agent

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload the Combined CSV for All Objectives", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file).fillna(0)

    for col in ["Success_Label", "Expected_Overload", "Risk_Flag"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col]).astype(int)

    st.success("✅ File uploaded successfully!")
    st.write(df.head())

    tabs = st.tabs([
        "1️⃣ Sprint Completion Forecast",
        "2️⃣ Workload Projection Forecast",
        "3️⃣ Time to Resolve Estimation",
        "4️⃣ Burnout Risk Alerts",
        "5️⃣ Resource Allocation Suggestions",
        "🤖 Agentic AI Supervisor"
    ])

    # --------------------------------------------------
    # OBJECTIVE 1
    # --------------------------------------------------
    with tabs[0]:
        X1 = df[['Planned_Story_Points_Sprint','Completed_Story_Points',
                 'Percent_Done','Days_Remaining_Sprint',
                 'Historical_Velocity','Blocked_Stories','Scope_Change']]
        y1 = df['Success_Label']

        sprint_model = LogisticRegression(max_iter=1000)
        sprint_model.fit(X1, y1)

        st.header("Sprint Completion Forecast")
        psp = st.number_input("Planned SP", 1, 100, 40)
        csp = st.number_input("Completed SP", 0, 100, 30)
        pdone = st.slider("% Done", 0.0, 100.0, 70.0)
        drs = st.number_input("Days Remaining", 0, 30, 5)
        hv = st.number_input("Historical Velocity", 0, 100, 35)
        bs = st.number_input("Blocked Stories", 0, 10, 1)
        sc = st.number_input("Scope Change", -20, 20, 0)

        if st.button("Predict Sprint Success"):
            features = np.array([[psp, csp, pdone, drs, hv, bs, sc]])
            sprint_prob = sprint_model.predict_proba(features)[0][1]
            st.metric("Sprint Success Probability", f"{sprint_prob:.2f}")

    # --------------------------------------------------
    # OBJECTIVE 2
    # --------------------------------------------------
    with tabs[1]:
        X2 = df[['Planned_Story_Points_Resource','Current_Assigned_SP',
                 'Historical_Avg_SP','Remaining_Days_Resource',
                 'High_Priority_Tasks_Resource','Current_Workload_Percent']]
        y2 = df['Expected_Overload']

        workload_model = RandomForestClassifier()
        workload_model.fit(X2, y2)

        st.header("Workload Projection")
        psp2 = st.number_input("Planned SP (Resource)", 1, 100, 35)
        casp = st.number_input("Current Assigned SP", 0, 100, 40)
        hasp = st.number_input("Historical Avg SP", 1, 100, 30)
        rdr = st.number_input("Remaining Days", 1, 30, 5)
        hpt = st.number_input("High Priority Tasks", 0, 10, 2)
        cwp = st.number_input("Current Workload %", 0, 200, 120)

        if st.button("Predict Overload"):
            features = np.array([[psp2, casp, hasp, rdr, hpt, cwp]])
            overload_prob = workload_model.predict_proba(features)[0][1]
            st.metric("Overload Probability", f"{overload_prob:.2f}")

    # --------------------------------------------------
    # OBJECTIVE 3
    # --------------------------------------------------
    with tabs[2]:
        X3 = pd.get_dummies(df[['Issue_Type','Priority']], drop_first=False)
        X3 = pd.concat([X3, df[['Original_Estimate_Hours','Story_Points_Issue']]], axis=1)
        y3 = df['Resolution_Time_Hours']

        ttr_model = LinearRegression()
        ttr_model.fit(X3, y3)

        st.header("Time to Resolve Estimation")
        itype = st.selectbox("Issue Type", ["Bug","Story","Task"])
        pr = st.selectbox("Priority", ["Low","Medium","High"])
        oe = st.number_input("Original Estimate", 1, 50, 8)
        sp = st.number_input("Story Points", 1, 20, 5)

        test_row = pd.DataFrame([{
            'Issue_Type_Bug': itype == "Bug",
            'Issue_Type_Story': itype == "Story",
            'Issue_Type_Task': itype == "Task",
            'Priority_Low': pr == "Low",
            'Priority_Medium': pr == "Medium",
            'Priority_High': pr == "High",
            'Original_Estimate_Hours': oe,
            'Story_Points_Issue': sp
        }]).reindex(columns=X3.columns, fill_value=0)

        if st.button("Estimate Resolution Time"):
            pred_time = max(0, ttr_model.predict(test_row)[0])
            st.metric("Estimated Resolution Time (hrs)", f"{pred_time:.1f}")

    # --------------------------------------------------
    # OBJECTIVE 4
    # --------------------------------------------------
    with tabs[3]:
        X4 = df[['Total_SP_This_Sprint','Historical_Avg_SP_Burnout',
                 'High_Priority_Tasks_Burnout','Consecutive_Overloads']]
        y4 = df['Risk_Flag']

        burnout_model = RandomForestClassifier()
        burnout_model.fit(X4, y4)

        st.header("Burnout Risk")
        tsp = st.number_input("Total SP", 0, 100, 40)
        hasp4 = st.number_input("Historical Avg SP", 1, 100, 25)
        hpt4 = st.number_input("High Priority Tasks", 0, 10, 2)
        co = st.number_input("Consecutive Overloads", 0, 5, 2)

        if st.button("Check Burnout Risk"):
            burnout_prob = burnout_model.predict_proba([[tsp, hasp4, hpt4, co]])[0][1]
            st.metric("Burnout Probability", f"{burnout_prob:.2f}")

    # --------------------------------------------------
    # OBJECTIVE 5
    # --------------------------------------------------
    with tabs[4]:
        le1, le2 = LabelEncoder(), LabelEncoder()
        df['Summary_enc'] = le1.fit_transform(df['Summary'].astype(str))
        df['Labels_enc'] = le2.fit_transform(df['Labels'].astype(str))

        X5 = df[['Summary_enc','Labels_enc','Original_Estimate_Resource','Story_Points_Resource']]
        y5 = df['Assignee_Resource']

        alloc_model = RandomForestClassifier()
        alloc_model.fit(X5, y5)

        st.header("Resource Allocation")
        summary = st.text_input("Summary", "Fix bug")
        label = st.text_input("Label", "Bug")
        oe5 = st.number_input("Original Estimate (Resource)", 1, 50, 8)
        sp5 = st.number_input("Story Points (Resource)", 1, 20, 5)

        try:
            s_enc = le1.transform([summary])[0]
            l_enc = le2.transform([label])[0]
        except:
            s_enc, l_enc = 0, 0

        if st.button("Suggest Assignee"):
            assignee = alloc_model.predict([[s_enc, l_enc, oe5, sp5]])[0]
            st.success(f"Recommended Assignee: {assignee}")

    # --------------------------------------------------
    # AGENTIC AI + GRAPHS
    # --------------------------------------------------
    with tabs[5]:
        st.header("🤖 Agentic AI Supervisor")

        sprint_risk = 1 - sprint_prob if 'sprint_prob' in locals() else 0.5
        overload_risk = overload_prob if 'overload_prob' in locals() else 0.5
        delay_risk = 0.5
        burnout_risk = burnout_prob if 'burnout_prob' in locals() else 0.5
        resource_risk = 0.4

        signals = {
            "sprint": sprint_risk,
            "overload": overload_risk,
            "delay": delay_risk,
            "burnout": burnout_risk,
            "resource": resource_risk
        }

        if st.button("Run Agent Decision"):
            risk, action = agent.decide(signals)
            st.metric("Overall Project Risk", f"{risk:.2f}")
            st.success(action)

        if agent.memory:
            st.subheader("📊 Objective Contribution")
            fig1, ax1 = plt.subplots()
            ax1.bar(signals.keys(), signals.values())
            st.pyplot(fig1)

            st.subheader("📈 Risk Trend Over Time")
            risks = [m["risk"] for m in agent.memory]
            fig2, ax2 = plt.subplots()
            ax2.plot(risks, marker="o")
            st.pyplot(fig2)

            st.subheader("🥧 Decision Distribution")
            actions = pd.Series([m["action"] for m in agent.memory]).value_counts()
            fig3, ax3 = plt.subplots()
            ax3.pie(actions, labels=actions.index, autopct="%1.1f%%")
            st.pyplot(fig3)
