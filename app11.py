import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AI Agile Dashboard", layout="wide")
st.title("🚀 AI Agile Project Management Dashboard")

uploaded_file = st.file_uploader("📁 Upload the Combined CSV for All Objectives", type="csv")

# -------------------------------------------------------------------
# SAFE DATA LOAD
# -------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna(0)
    for col in ['Success_Label', 'Expected_Overload', 'Risk_Flag']:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(df[col]).astype(int)
    return df

# -------------------------------------------------------------------
# MODEL TRAINING FUNCTIONS
# -------------------------------------------------------------------

@st.cache_resource
def train_sprint_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


@st.cache_resource
def train_workload_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model


@st.cache_resource
def train_ttr_model(X, y):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Issue_Type", "Priority"]),
        ("num", "passthrough", ["Original_Estimate_Hours", "Story_Points_Issue"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", LinearRegression())
    ])
    model.fit(X, y)
    return model


@st.cache_resource
def train_burnout_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model


@st.cache_resource
def train_allocation_model(X, y):

    preprocessor = ColumnTransformer([
        ("summary", TfidfVectorizer(), "Summary"),
        ("labels", TfidfVectorizer(), "Labels"),
        ("num", "passthrough",
         ["Original_Estimate_Resource", "Story_Points_Resource"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
if uploaded_file:

    df = load_data(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write(df.head())

    tabs = st.tabs([
        "1️⃣ Sprint Completion Forecast",
        "2️⃣ Workload Projection Forecast",
        "3️⃣ Time to Resolve Estimation",
        "4️⃣ Burnout Risk Alerts",
        "5️⃣ Resource Allocation Suggestions"
    ])

# -------------------------------------------------------------------
# OBJECTIVE 1
# -------------------------------------------------------------------
    with tabs[0]:
        st.header("📌 Sprint Completion Forecast")

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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_sprint_model(X_train, y_train)

            st.write("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

            st.subheader("🔍 Predict Sprint Success")

            input_df = pd.DataFrame([{
                'Planned_Story_Points_Sprint': st.number_input("Planned SP", 1, 100, 40),
                'Completed_Story_Points': st.number_input("Completed SP", 0, 100, 30),
                'Percent_Done': st.slider("% Done", 0.0, 100.0, 75.0),
                'Days_Remaining_Sprint': st.number_input("Days Remaining", 0, 30, 5),
                'Historical_Velocity': st.number_input("Historical Velocity", 0, 100, 35),
                'Blocked_Stories': st.number_input("Blocked Stories", 0, 10, 1),
                'Scope_Change': st.number_input("Scope Change", -20, 20, 0)
            }])

            if st.button("Predict Sprint Success"):
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
                if pred:
                    st.success(f"✅ Likely to Complete ({prob:.2f})")
                else:
                    st.warning(f"⚠️ Risk of Spillover ({prob:.2f})")

# -------------------------------------------------------------------
# OBJECTIVE 2
# -------------------------------------------------------------------
    with tabs[1]:
        st.header("📌 Workload Projection Forecast")

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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_workload_model(X_train, y_train)

            st.write("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

            input_df = pd.DataFrame([{
                'Planned_Story_Points_Resource': st.number_input("Planned SP", 1, 100, 35),
                'Current_Assigned_SP': st.number_input("Current Assigned SP", 0, 100, 40),
                'Historical_Avg_SP': st.number_input("Historical Avg SP", 1, 100, 30),
                'Remaining_Days_Resource': st.number_input("Remaining Days", 1, 30, 5),
                'High_Priority_Tasks_Resource': st.number_input("High Priority Tasks", 0, 10, 2),
                'Current_Workload_Percent': st.number_input("Workload %", 0, 200, 125)
            }])

            if st.button("Predict Overload"):
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
                if pred:
                    st.warning(f"⚠️ Overload Risk ({prob:.2f})")
                else:
                    st.success(f"✅ Within Capacity ({prob:.2f})")

# -------------------------------------------------------------------
# OBJECTIVE 3
# -------------------------------------------------------------------
    with tabs[2]:
        st.header("📌 Time to Resolve Estimation")

        X = df[['Issue_Type', 'Priority', 'Original_Estimate_Hours', 'Story_Points_Issue']]
        y = df['Resolution_Time_Hours']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_ttr_model(X_train, y_train)

        st.write("MSE:", mean_squared_error(y_test, model.predict(X_test)))

        input_df = pd.DataFrame([{
            "Issue_Type": st.selectbox("Issue Type", ['Bug', 'Story', 'Task']),
            "Priority": st.selectbox("Priority", ['Low', 'Medium', 'High']),
            "Original_Estimate_Hours": st.number_input("Original Estimate", 1, 50, 8),
            "Story_Points_Issue": st.number_input("Story Points", 1, 20, 5)
        }])

        if st.button("Estimate Resolution Time"):
            pred = model.predict(input_df)[0]
            st.info(f"⏰ Estimated Resolution Time: {max(0, pred):.1f} hours")

# -------------------------------------------------------------------
# OBJECTIVE 4
# -------------------------------------------------------------------
    with tabs[3]:
        st.header("📌 Burnout Risk Alerts")

        X = df[[
            'Total_SP_This_Sprint',
            'Historical_Avg_SP_Burnout',
            'High_Priority_Tasks_Burnout',
            'Consecutive_Overloads'
        ]]
        y = df['Risk_Flag']

        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_burnout_model(X_train, y_train)

            st.write("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

            input_df = pd.DataFrame([{
                'Total_SP_This_Sprint': st.number_input("Total SP", 0, 100, 40),
                'Historical_Avg_SP_Burnout': st.number_input("Historical Avg SP", 1, 100, 25),
                'High_Priority_Tasks_Burnout': st.number_input("High Priority Tasks", 0, 10, 2),
                'Consecutive_Overloads': st.number_input("Consecutive Overloads", 0, 5, 2)
            }])

            if st.button("Check Burnout Risk"):
                pred = model.predict(input_df)[0]
                if pred:
                    st.warning("⚠️ Burnout Risk Detected!")
                else:
                    st.success("✅ Workload looks healthy!")

# -------------------------------------------------------------------
# OBJECTIVE 5
# -------------------------------------------------------------------
    with tabs[4]:
        st.header("📌 Resource Allocation Suggestions")

        X = df[['Summary', 'Labels', 'Original_Estimate_Resource', 'Story_Points_Resource']]
        y = df['Assignee_Resource']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_allocation_model(X_train, y_train)

        st.write("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

        input_df = pd.DataFrame([{
            "Summary": st.text_input("Summary", "Fix login bug"),
            "Labels": st.text_input("Label", "Bug"),
            "Original_Estimate_Resource": st.number_input("Original Estimate", 1, 50, 8),
            "Story_Points_Resource": st.number_input("Story Points", 1, 20, 5)
        }])

        if st.button("Suggest Assignee"):
            assignee = model.predict(input_df)[0]
            st.success(f"✅ Recommended Assignee: {assignee}")
