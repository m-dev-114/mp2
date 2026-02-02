import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Agile Dashboard", layout="wide")

st.title("🚀 AI Agile Project Management Dashboard")

uploaded_file = st.file_uploader("📁 Upload the Combined CSV for All Objectives", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.fillna(0) 
    for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(df[col]).astype(int)

    st.success("✅ File uploaded successfully!")
    st.write(df.head())

    tabs = st.tabs([
        "1️⃣ Sprint Completion Forecast",
        "2️⃣ Workload Projection Forecast",
        "3️⃣ Time to Resolve Estimation",
        "4️⃣ Burnout Risk Alerts",
        "5️⃣ Resource Allocation Suggestions"
    ])

# Objective 1: Sprint Completion Forecast
    with tabs[0]:
        st.header("📌 Objective 1 — Sprint Completion Forecasting")

        try:
            X1 = df[[
                'Planned_Story_Points_Sprint',
                'Completed_Story_Points',
                'Percent_Done',
                'Days_Remaining_Sprint',
                'Historical_Velocity',
                'Blocked_Stories',
                'Scope_Change'
            ]]
            y1 = df['Success_Label']

            if len(y1.unique()) > 1:
                X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

                sprint_model = LogisticRegression(max_iter=1000)
                sprint_model.fit(X1_train, y1_train)

                y1_pred = sprint_model.predict(X1_test)
                st.write(f"✅ Accuracy: {accuracy_score(y1_test, y1_pred):.2f}")
                st.text(classification_report(y1_test, y1_pred))

                st.subheader("🔍 Predict Sprint Success")
                psp = st.number_input("Planned Story Points", 1, 100, 40, key="obj1_psp")
                csp = st.number_input("Completed Story Points", 0, 100, 30, key="obj1_csp")
                percent_done = st.slider("% Done", 0.0, 100.0, 75.0, key="obj1_pd")
                drs = st.number_input("Days Remaining", 0, 30, 5, key="obj1_drs")
                hv = st.number_input("Historical Velocity", 0, 100, 35, key="obj1_hv")
                bs = st.number_input("Blocked Stories", 0, 10, 1, key="obj1_bs")
                sc = st.number_input("Scope Change", -20, 20, 0, key="obj1_sc")

                if st.button("Predict Sprint Success", key="obj1_btn"):
                    features = np.array([[psp, csp, percent_done, drs, hv, bs, sc]])
                    p = sprint_model.predict(features)[0]
                    prob = sprint_model.predict_proba(features)[0][1]
                    if p:
                        st.success(f"✅ Likely to Complete! ({prob:.2f})")
                    else:
                        st.warning(f"⚠️ Risk of Spillover! ({prob:.2f})")
            else:
                st.error("⚠️ Not enough class variety in Success_Label column.")
        except Exception as e:
            st.error(f"Error in Objective 1: {e}")

# ------------------------------------------------------------------
# Objective 2: Workload Projection
    with tabs[1]:
        st.header("📌 Objective 2 — Workload Projection Forecast")
        try:
            X2 = df[[
                'Planned_Story_Points_Resource',
                'Current_Assigned_SP',
                'Historical_Avg_SP',
                'Remaining_Days_Resource',
                'High_Priority_Tasks_Resource',
                'Current_Workload_Percent'
            ]]
            y2 = df['Expected_Overload']

            if len(y2.unique()) > 1:
                X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

                workload_model = RandomForestClassifier()
                workload_model.fit(X2_train, y2_train)

                y2_pred = workload_model.predict(X2_test)
                st.write(f"✅ Accuracy: {accuracy_score(y2_test, y2_pred):.2f}")
                st.text(classification_report(y2_test, y2_pred))

                st.subheader("🔍 Predict Overload Risk")
                psp2 = st.number_input("Planned SP", 1, 100, 35, key="obj2_psp2")
                casp = st.number_input("Current Assigned SP", 0, 100, 40, key="obj2_casp")
                hasp = st.number_input("Historical Avg SP", 1, 100, 30, key="obj2_hasp")
                rdr = st.number_input("Remaining Days", 1, 30, 5, key="obj2_rdr")
                hpt = st.number_input("High Priority Tasks", 0, 10, 2, key="obj2_hpt")
                cwp = st.number_input("Current Workload %", 0, 200, 125, key="obj2_cwp")

                if st.button("Predict Overload", key="obj2_btn"):
                    features = np.array([[psp2, casp, hasp, rdr, hpt, cwp]])
                    pred = workload_model.predict(features)[0]
                    prob = workload_model.predict_proba(features)[0][1]
                    if pred:
                        st.warning(f"⚠️ Overload Risk! ({prob:.2f})")
                    else:
                        st.success(f"✅ Within Capacity ({prob:.2f})")
            else:
                st.error("⚠️ Not enough class variety in Expected_Overload column.")
        except Exception as e:
            st.error(f"Error in Objective 2: {e}")

# ------------------------------------------------------------------
# Objective 3: Time to Resolve Estimation
    with tabs[2]:
        st.header("📌 Objective 3 — Time to Resolve Estimation")
        try:
            X3 = pd.get_dummies(df[['Issue_Type', 'Priority']], drop_first=False)
            X3 = pd.concat([X3, df[['Original_Estimate_Hours', 'Story_Points_Issue']]], axis=1)
            y3 = df['Resolution_Time_Hours']

            X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

            ttr_model = LinearRegression()
            ttr_model.fit(X3_train, y3_train)

            y3_pred = ttr_model.predict(X3_test)
            mse = mean_squared_error(y3_test, y3_pred)
            st.write(f"✅ MSE: {mse:.2f}")

            st.subheader("🔍 Estimate Time to Resolve")
            issue_type = st.selectbox("Issue Type", ['Bug', 'Story', 'Task'], key="obj3_it")
            priority = st.selectbox("Priority", ['Low', 'Medium', 'High'], key="obj3_pri")
            oe = st.number_input("Original Estimate", 1, 50, 8, key="obj3_oe")
            sp = st.number_input("Story Points", 1, 20, 5, key="obj3_sp")

            # Build feature row and align with training
            test_row = pd.DataFrame([{
                'Issue_Type_Bug': 1 if issue_type == 'Bug' else 0,
                'Issue_Type_Story': 1 if issue_type == 'Story' else 0,
                'Issue_Type_Task': 1 if issue_type == 'Task' else 0,
                'Priority_Low': 1 if priority == 'Low' else 0,
                'Priority_Medium': 1 if priority == 'Medium' else 0,
                'Priority_High': 1 if priority == 'High' else 0,
                'Original_Estimate_Hours': oe,
                'Story_Points_Issue': sp
            }])

            # Align columns
            test_row = test_row.reindex(columns=X3.columns, fill_value=0)

            if st.button("Estimate Resolution Time", key="obj3_btn"):
                pred_time = ttr_model.predict(test_row)[0]
                pred_time = max(0, pred_time)  # avoid negative
                st.info(f"⏰ Estimated Resolution Time: {pred_time:.1f} hours")
        except Exception as e:
            st.error(f"Error in Objective 3: {e}")

# ------------------------------------------------------------------
# Objective 4: Burnout Risk Alerts
    with tabs[3]:
        st.header("📌 Objective 4 — Burnout Risk Alerts")
        try:
            X4 = df[[
                'Total_SP_This_Sprint',
                'Historical_Avg_SP_Burnout',
                'High_Priority_Tasks_Burnout',
                'Consecutive_Overloads'
            ]]
            y4 = df['Risk_Flag']

            if len(y4.unique()) > 1:
                X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)

                burnout_model = RandomForestClassifier()
                burnout_model.fit(X4_train, y4_train)

                y4_pred = burnout_model.predict(X4_test)
                st.write(f"✅ Accuracy: {accuracy_score(y4_test, y4_pred):.2f}")
                st.text(classification_report(y4_test, y4_pred))

                st.subheader("🔍 Check Burnout Risk")
                tsp = st.number_input("Total SP This Sprint", 0, 100, 40, key="obj4_tsp")
                hasp4 = st.number_input("Historical Avg SP", 1, 100, 25, key="obj4_hasp4")
                hpt4 = st.number_input("High Priority Tasks", 0, 10, 2, key="obj4_hpt4")
                co = st.number_input("Consecutive Overloads", 0, 5, 2, key="obj4_co")

                if st.button("Check Burnout Risk", key="obj4_btn"):
                    pred = burnout_model.predict([[tsp, hasp4, hpt4, co]])[0]
                    if pred:
                        st.warning("⚠️ Burnout Risk Detected!")
                    else:
                        st.success("✅ Workload looks healthy!")
            else:
                st.error("⚠️ Not enough class variety in Risk_Flag column.")
        except Exception as e:
            st.error(f"Error in Objective 4: {e}")

# ------------------------------------------------------------------
# Objective 5: Resource Allocation
    with tabs[4]:
        st.header("📌 Objective 5 — Resource Allocation Suggestions")
        try:
            # Encode free-text columns with LabelEncoder instead of one-hot
            le_summary = LabelEncoder()
            le_labels = LabelEncoder()

            df['Summary_enc'] = le_summary.fit_transform(df['Summary'].astype(str))
            df['Labels_enc'] = le_labels.fit_transform(df['Labels'].astype(str))

            X5 = df[['Summary_enc', 'Labels_enc', 'Original_Estimate_Resource', 'Story_Points_Resource']]
            y5 = df['Assignee_Resource']

            X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=42)

            alloc_model = RandomForestClassifier()
            alloc_model.fit(X5_train, y5_train)

            y5_pred = alloc_model.predict(X5_test)
            st.write(f"✅ Accuracy: {accuracy_score(y5_test, y5_pred):.2f}")

            st.subheader("🔍 Suggest Assignee")
            summary = st.text_input("Summary (short description)", "Fix bug")
            label = st.text_input("Label (category)", "Bug")
            oe5 = st.number_input("Original Estimate", 1, 50, 8, key="obj5_oe")
            sp5 = st.number_input("Story Points", 1, 20, 5, key="obj5_sp")

            # Encode with same encoders (handle unseen with try/except)
            try:
                summary_enc = le_summary.transform([summary])[0]
            except:
                summary_enc = 0
            try:
                label_enc = le_labels.transform([label])[0]
            except:
                label_enc = 0

            test_row = pd.DataFrame([{
                'Summary_enc': summary_enc,
                'Labels_enc': label_enc,
                'Original_Estimate_Resource': oe5,
                'Story_Points_Resource': sp5
            }])

            if st.button("Suggest Assignee", key="obj5_btn"):
                assignee = alloc_model.predict(test_row)[0]
                st.success(f"✅ Recommended Assignee: {assignee}")
        except Exception as e:
            st.error(f"Error in Objective 5: {e}")
