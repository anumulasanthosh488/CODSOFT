import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="metric-container"]{
    background:#1F2937;
    border-radius:15px;
    padding:18px;
    border:1px solid #2E3A46;
}

h1,h2,h3,h4{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("📈 Sales Prediction Dashboard")

st.caption(
    "AI Powered Sales Analytics using Simple Linear Regression"
)

st.divider()

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Advertising Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    # ------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------

    X = df[["TV"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.session_state["model"] = model

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)

    mse = mean_squared_error(y_test,y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test,y_pred)

    st.success("✅ Model Trained Successfully")

    # ---------------------------------------------
    # DASHBOARD VALUES
    # ---------------------------------------------

    total_records=len(df)

    avg_tv=df["TV"].mean()

    avg_sales=df["Sales"].mean()

    # ==========================================================
# SIDEBAR
# ==========================================================

    with st.sidebar:

        st.image(
            "https://img.icons8.com/color/96/combo-chart--v1.png",
            width=70
        )

        st.title("Sales Dashboard")

        st.markdown("---")

        st.subheader("📌 Navigation")

        st.write("🏠 Dashboard")
        st.write("📊 Analytics")
        st.write("🎯 Prediction")
        st.write("📈 Model Performance")
        st.write("ℹ️ About")

        st.markdown("---")

        st.subheader("📁 Project Details")

        st.info("Algorithm : Simple Linear Regression")

        st.success(f"Rows : {len(df)}")

        st.success(f"Columns : {df.shape[1]}")

        st.info("Dataset : Advertising.csv")

        st.warning("Developer : A. Santhosh")

        st.success("Department : CSE (Data Science)")

    # ==========================================================
    # DASHBOARD OVERVIEW
    # ==========================================================

    st.markdown("---")

    st.subheader("📊 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📄 Total Records",
            total_records
        )

    with c2:
        st.metric(
            "📺 Avg TV Budget",
            f"{avg_tv:.2f}"
        )

    with c3:
        st.metric(
            "💰 Avg Sales",
            f"{avg_sales:.2f}"
        )

    with c4:
        st.metric(
            "🎯 Accuracy",
            f"{r2*100:.2f}%"
        )

        # ==========================================================
    # DATASET PREVIEW
    # ==========================================================

    st.markdown("---")

    st.subheader("📂 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # CHARTS
    # ==========================================================

    st.subheader("📈 Sales Analytics")

    left_chart, right_chart = st.columns([2, 1])

    # ==========================================================
    # LEFT COLUMN
    # ==========================================================

    with left_chart:

        st.subheader("📈 TV Advertising vs Sales")

        fig1 = px.scatter(
            df,
            x="TV",
            y="Sales",
            color="Sales",
            size="Sales",
            hover_data=["Radio", "Newspaper"],
            title="TV Advertising vs Sales",
            color_continuous_scale="Viridis"
        )

        # Regression Line
        x_line = np.linspace(df["TV"].min(), df["TV"].max(), 100)
        y_line = model.predict(pd.DataFrame({"TV": x_line}))

        fig1.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regression Line",
                line=dict(color="red", width=3)
            )
        )

        fig1.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig1, use_container_width=True)

    # ==========================================================
    # RIGHT COLUMN
    # ==========================================================

    with right_chart:

        st.subheader("📊 Sales Distribution")

        fig2 = px.histogram(
            df,
            x="Sales",
            nbins=20,
            color_discrete_sequence=["orange"]
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ==========================================================
    # CORRELATION MATRIX
    # ==========================================================

    st.subheader("🔥 Correlation Matrix")

    corr = df.corr(numeric_only=True)

    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    fig3.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ==========================================================
    # AVERAGE ADVERTISING BUDGET
    # ==========================================================

    st.subheader("📊 Average Advertising Budget")

    avg_df = pd.DataFrame({
        "Advertising": ["TV", "Radio", "Newspaper"],
        "Average Budget": [
            df["TV"].mean(),
            df["Radio"].mean(),
            df["Newspaper"].mean()
        ]
    })

    fig4 = px.bar(
        avg_df,
        x="Advertising",
        y="Average Budget",
        color="Advertising",
        text_auto=".2f"
    )

    fig4.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig4, use_container_width=True)

    # ====================================
# SALES PREDICTION
# ====================================

st.divider()

st.header("🎯 Sales Prediction")

st.write("Predict expected sales based on TV advertising budget.")

tv_budget = st.slider(
    "TV Advertising Budget ($1000)",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=1.0
)

st.metric("Entered Budget", f"${tv_budget:.2f}K")

if st.button("🚀 Predict Sales", use_container_width=True):

    prediction = st.session_state["model"].predict([[tv_budget]])[0]

    st.success(f"📈 Predicted Sales: **{prediction:.2f} Units**")

    if prediction > 20:
        st.balloons()
        st.success("Excellent! Expected sales are very high.")
    elif prediction > 10:
        st.info("Good sales are expected.")
    else:
        st.warning("Sales prediction is low. Consider increasing advertising budget.")

     
    
    st.header("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")

    st.success("Model trained successfully using Simple Linear Regression.")   

    
    st.header("ℹ️ About This Project")

    st.write("""
This project predicts sales using TV advertising budget.

🔹 Algorithm: Simple Linear Regression
🔹 Dataset: Advertising.csv
🔹 Libraries:
• Streamlit
• Pandas
• NumPy
• Scikit-learn
• Plotly

Developed as an AI-powered Sales Prediction Dashboard.
""")