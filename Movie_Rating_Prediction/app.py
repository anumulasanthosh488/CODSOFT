import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load Model & Encoders
# ----------------------------

model = joblib.load("movie_rating_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(
    page_title="Movie Rating Prediction",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("🎬 Movie Rating Predictor")
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
    width=180
)
st.sidebar.markdown("---")

st.sidebar.header("📋 Project Information")
st.sidebar.write("Machine Learning Internship Project")

st.sidebar.header("🤖 Algorithm")
st.sidebar.success("Random Forest Regressor")

st.sidebar.header("📊 Dataset")
st.sidebar.info("IMDb Movies India")

st.sidebar.header("📈 Model Performance")
st.sidebar.success("MAE : 0.826")
st.sidebar.success("RMSE : 1.095")
st.sidebar.success("R² Score : 0.355")

st.sidebar.markdown("---")
st.sidebar.write("Developed using")
st.sidebar.write("✔ Python")
st.sidebar.write("✔ Pandas")
st.sidebar.write("✔ Scikit-Learn")
st.sidebar.write("✔ Streamlit")
# ===============================
# Main Title
# ===============================

st.title("🎬 Movie Rating Prediction System")
st.markdown("Predict the IMDb rating of a movie using Machine Learning.")

st.markdown("---")

left, right = st.columns([3, 2])

# ===============================
# Left Column
# ===============================

with left:

    st.subheader("📝 Enter Movie Details")

    year = st.selectbox(
        "📅 Release Year",
        list(range(1930, 2026)),
        index=len(list(range(1930, 2026))) - 6
    )

    st.subheader("⏱ Movie Duration")

    col1, col2 = st.columns(2)

    with col1:
        hours = st.selectbox(
            "Hours",
            [0, 1, 2, 3, 4],
            index=2
        )

    with col2:
        minutes = st.selectbox(
            "Minutes",
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            index=6
        )

    duration = hours * 60 + minutes

    votes = st.number_input(
        "👍 IMDb Votes",
        min_value=0,
        value=5000,
        step=100
    )

    genre = st.selectbox(
        "🎭 Genre",
        sorted(encoders["Genre"].classes_)
    )

    director = st.selectbox(
        "🎬 Director",
        sorted(encoders["Director"].classes_)
    )

    actor1 = st.selectbox(
        "⭐ Lead Actor",
        sorted(encoders["Actor 1"].classes_)
    )

    actor2 = st.selectbox(
        "⭐ Supporting Actor",
        sorted(encoders["Actor 2"].classes_)
    )

    actor3 = st.selectbox(
        "⭐ Supporting Actor 2",
        sorted(encoders["Actor 3"].classes_)
    )

    # ===============================
# Right Column
# ===============================

with right:

    st.subheader("🎯 Prediction")

    if st.button(
    "🎯 Predict IMDb Rating",
    type="primary",
    use_container_width=True
    ):
        # Encode categorical values
        genre_code = encoders["Genre"].transform([genre])[0]
        director_code = encoders["Director"].transform([director])[0]
        actor1_code = encoders["Actor 1"].transform([actor1])[0]
        actor2_code = encoders["Actor 2"].transform([actor2])[0]
        actor3_code = encoders["Actor 3"].transform([actor3])[0]

        # Create input for model
        input_data = pd.DataFrame(
            [[
                year,
                duration,
                votes,
                genre_code,
                director_code,
                actor1_code,
                actor2_code,
                actor3_code
            ]],
            columns=[
                "Year",
                "Duration",
                "Votes",
                "Genre",
                "Director",
                "Actor 1",
                "Actor 2",
                "Actor 3"
            ]
        )

        # Prediction
        prediction = model.predict(input_data)
        rating = round(float(prediction[0]), 2)

        st.metric(
            label="⭐ Predicted IMDb Rating",
            value=f"{rating:.2f}/10"
        )

        

        if rating >= 8:
            st.balloons()
            st.success("🏆 Excellent Movie")
        elif rating >= 7:
            st.success("👍 Very Good Movie")
        elif rating >= 5:
            st.warning("🙂 Average Movie")
        else:
            st.error("👎 Below Average Movie")

        st.markdown("---")


        st.subheader("📋 Movie Summary")

        st.write(f"**📅 Release Year:** {year}")
        st.write(f"**⏱ Duration:** {hours} hr {minutes} min")
        st.write(f"**👍 Votes:** {votes:,}")
        st.write(f"**🎭 Genre:** {genre}")
        st.write(f"**🎬 Director:** {director}")
        st.write(f"**⭐ Lead Actor:** {actor1}")
        st.write(f"**⭐ Supporting Actor:** {actor2}")
        st.write(f"**⭐ Supporting Actor 2:** {actor3}")