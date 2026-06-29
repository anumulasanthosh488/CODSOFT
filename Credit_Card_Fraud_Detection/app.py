import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("fraud_model.pkl")

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main{
    background:#f4f7fc;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0B1220,#162033);
    color:white;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.metric-box{
    border-radius:15px;
    padding:20px;
    color:white;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.blue{
background:#2563EB;
}

.green{
background:#16A34A;
}

.red{
background:#DC2626;
}

.orange{
background:#F59E0B;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,.12);
}

h1{
font-weight:700;
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.markdown("""
    # 💳 Fraud Detection

    AI Powered Analytics
    """)

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Prediction",
            "📊 Performance",
            "ℹ About Project"
        ]
    )

    st.markdown("---")

    st.markdown("""
### 📄 Project Info

**Project**

Credit Card Fraud Detection

**Algorithm**

Random Forest

**Developer**

A. Santhosh

**Branch**

Computer Science & Engineering
(Data Science)

**Technologies**

• Python

• Streamlit

• Plotly

• Pandas

• Scikit-Learn

• Joblib
""")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<div style="
background:linear-gradient(135deg,#1e293b,#0f172a);
padding:30px;
border-radius:18px;
box-shadow:0px 5px 20px rgba(0,0,0,.25);
text-align:center;
margin-bottom:20px;
">

<h1 style="color:white;">
💳 Credit Card Fraud Detection Dashboard
</h1>

<p style="font-size:20px;color:#CBD5E1;">
AI Powered Fraud Analytics using Random Forest
</p>

</div>
""", unsafe_allow_html=True)

col1,col2,col3,col4=st.columns(4)

with col1:
    st.info("🤖 Model\n\nRandom Forest")

with col2:
    st.success("📊 Dataset\n\n284,807 Records")

with col3:
    st.warning("🎯 Accuracy\n\n99.94%")

with col4:
    st.error("🚨 Fraud Detection\n\nReal-Time")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

if menu == "🏠 Dashboard":

    uploaded_file = st.file_uploader(
        "📂 Upload Credit Card Transaction Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:


        st.success("✅ Model Loaded Successfully")

        
        st.markdown("### Project Highlights")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.info("""
            **Real-Time Detection**

            Instant prediction using trained model.
            """)

        with c2:
            st.success("""
            🤖 **Random Forest**

            Machine Learning Algorithm
            """)

        with c3:
            st.warning("""
            📊 **99.94% Accuracy**

            Highly Accurate Model
            """)

        with c4:
            st.error("""
            **Secure Analysis**

            Fast & Reliable Detection
            """)
        
        
        data = pd.read_csv(uploaded_file)

        if "Class" in data.columns:
            data = data.drop("Class",axis=1)

        with st.expander("📄 View Uploaded Dataset"):
            st.dataframe(data.head(20),use_container_width=True)

        # ------------------------------------------
        # MODEL PERFORMANCE
        # ------------------------------------------

        st.markdown("---")
        st.subheader("🤖 Model Performance")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", "99.94%")
        c2.metric("Precision", "83.54%")
        c3.metric("Recall", "82.65%")
        c4.metric("F1 Score", "83.09%")
        c5.metric("ROC-AUC", "99.92%")

        st.markdown("---")

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

    
        # ------------------------------------------
        # PREDICT BUTTON
        # ------------------------------------------

        if st.button("🚀 Predict Fraud Transactions"):
            st.session_state["prediction_data"] = data.copy()

            start_time = time.time()

            prediction = model.predict(data)

            prediction_proba = model.predict_proba(data)

            confidence = prediction_proba.max(axis=1) * 100

            data["Confidence (%)"] = confidence.round(2)

            data["Prediction"] = prediction

            data["Prediction"] = data["Prediction"].map({
                0:"Normal",
                1:"Fraud"
            })

            st.session_state["prediction_data"] = data.copy()

            fraud_count = (data["Prediction"]=="Fraud").sum()

            normal_count = (data["Prediction"]=="Normal").sum()

            total_transactions = len(data)

            fraud_rate = (fraud_count/total_transactions)*100

            st.markdown("---")

            processing_time = time.time() - start_time

            st.success(f"⚡ Prediction completed in {processing_time:.3f} seconds")

            st.subheader("📈 Prediction Summary")

            avg_confidence = confidence.mean()

            st.info(f"""
            ### 🤖 AI Prediction Confidence

            **Average Confidence:** {avg_confidence:.2f}%

            The Random Forest model is highly confident in its predictions.
            """)

            a,b,c,d = st.columns(4)

            with a:
                st.markdown(f"""
                <div class="metric-box blue">
                <h4>Total Transactions</h4>
                <h1>{total_transactions}</h1>
                </div>
                """,unsafe_allow_html=True)

            with b:
                st.markdown(f"""
                <div class="metric-box green">
                <h4>Normal Transactions</h4>
                <h1>{normal_count}</h1>
                </div>
                """,unsafe_allow_html=True)

            with c:
                st.markdown(f"""
                <div class="metric-box red">
                <h4>Fraud Transactions</h4>
                <h1>{fraud_count}</h1>
                </div>
                """,unsafe_allow_html=True)

            with d:
                st.markdown(f"""
                <div class="metric-box orange">
                    <h4>Fraud Rate</h4>
                    <h1>{fraud_rate:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                "Type":["Normal","Fraud"],
                "Count":[normal_count,fraud_count]
            })

            st.markdown("---")
            st.subheader("📊 Fraud Analytics Dashboard")

            left, right = st.columns(2)

            # -------------------------------
            # Doughnut Chart
            # -------------------------------

            with left:

                pie = px.pie(
                chart_data,
                names="Type",
                values="Count",
                hole=0.65,
                color="Type",
                color_discrete_map={
                    "Normal": "#4F8EF7",
                    "Fraud": "#FF4B4B"
                },
                title="Fraud vs Normal Transactions"
            )

            pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=16
            )

            pie.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            )

            st.plotly_chart(pie, use_container_width=True)
            # -------------------------------
            # Bar Chart
            # -------------------------------

            with right:

                bar = px.bar(
                chart_data,
                x="Type",
                y="Count",
                color="Type",
                text="Count",
                color_discrete_map={
                    "Normal": "#4F8EF7",
                    "Fraud": "#FF4B4B"
                },
                title="Transaction Comparison"
            )

            bar.update_traces(
                textposition="outside"
            )

            bar.update_layout(
                height=430,
                xaxis_title="Transaction Type",
                yaxis_title="Count",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(bar, use_container_width=True)
            # -------------------------------
            # Fraud Rate Gauge
            # -------------------------------

            st.subheader("🎯 Fraud Risk Indicator")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fraud_rate,
                    number={"suffix":" %"},
                    title={"text":"Fraud Rate"},
                    gauge={
                        "axis":{"range":[0,100]},
                        "bar":{"color":"crimson"},
                        "steps":[
                            {"range":[0,25],"color":"#BBF7D0"},
                            {"range":[25,50],"color":"#FDE68A"},
                            {"range":[50,100],"color":"#FECACA"}
                        ]
                    }
                )
            )

            gauge.update_layout(height=350)

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            # ------------------------------------------
            # Prediction Results
            # ------------------------------------------
        elif menu == "🔍 Prediction":

            st.markdown("---")
            st.subheader("📄 Prediction Results")

            st.markdown("---")
            st.subheader("🔍 Filter Prediction Results")

            if "prediction_data" not in st.session_state:
                st.warning("⚠️ Please go to Dashboard and click '🚀 Predict Fraud Transactions' first.")
                st.stop()

            data = st.session_state.get("prediction_data", None)

            if data is None:
                st.warning("Please upload the dataset and click Predict first.")
                st.stop()

            st.write(data.columns)
            st.write(data["Prediction"].head())

            filter_option = st.selectbox(
                
                "Select Transaction Type",
                ["All Transactions", "Fraud Only", "Normal Only"]
            )
            st.write("Selected:", filter_option)
            st.write(data["Prediction"].value_counts())

            if filter_option == "Fraud Only":
                filtered_data = data[data["Prediction"] == "Fraud"]

            elif filter_option == "Normal Only":
                filtered_data = data[data["Prediction"] == "Normal"]

            else:
                filtered_data = data

            
            st.dataframe(
            filtered_data,
                use_container_width=True,
                height=400
            )

            # ------------------------------------------
            # Download CSV
            # ------------------------------------------

            csv = filtered_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="fraud_prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -------------------------------------------------
    # PERFORMANCE PAGE
    # -------------------------------------------------

    elif menu == "📊 Performance":

        st.header("📊 Model Performance")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", "99.94%")
        c2.metric("Precision", "83.54%")
        c3.metric("Recall", "82.65%")
        c4.metric("F1 Score", "83.09%")
        c5.metric("ROC-AUC", "99.92%")

        st.markdown("---")

        st.subheader("📊 Confusion Matrix")

        confusion = pd.DataFrame(
            [[56848,16],
            [17,81]],
            columns=["Predicted Normal","Predicted Fraud"],
            index=["Actual Normal","Actual Fraud"]
        )

        st.dataframe(confusion, use_container_width=True)

    # -------------------------------------------------
    # ABOUT PAGE
    # -------------------------------------------------

    elif menu == "ℹ About Project":

        st.header("📖 About Project")

        st.markdown("""
    ### 💳 Credit Card Fraud Detection using Machine Learning

    This project detects fraudulent credit card transactions using a
    **Random Forest Classifier**.

    ### 🎯 Objective

    To identify fraudulent transactions accurately and help financial
    institutions reduce fraud-related losses.

    ### 🚀 Features

    - Upload CSV Dataset
    - Fraud Prediction
    - KPI Dashboard
    - Interactive Charts
    - Fraud Gauge
    - Confusion Matrix
    - Download Prediction Results

    ### 🛠 Technologies

    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Scikit-learn
    - Joblib

    ### 🤖 Machine Learning

    Random Forest Classifier

    ### 👨‍💻 Developer

    **A. Santhosh**

    **Branch:** Computer Science & Engineering (Data Science)
    """)

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
    <div style="text-align:center;color:gray;">
    © 2026 Credit Card Fraud Detection Dashboard<br>
    Developed by <b>A. Santhosh</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("⭐ Top 10 Important Features")

    if hasattr(model, "feature_importances_"):

        feature_columns = [
        col for col in data.columns
    if col not in ["Prediction", "Confidence (%)"]
    ]

        feature_importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="Top 10 Features Used by Random Forest"
        )

        fig.update_layout(
            height=500,
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    