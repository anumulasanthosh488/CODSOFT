# 🎬 Movie Rating Prediction System

A Machine Learning web application that predicts IMDb movie ratings based on movie details such as release year, duration, IMDb votes, genre, director, and cast.

This project was developed using **Python**, **Scikit-Learn**, and **Streamlit** as part of a **Machine Learning Internship**.

---

# 📌 Project Overview

The Movie Rating Prediction System uses a **Random Forest Regressor** model trained on the IMDb India Movies dataset to estimate the IMDb rating of a movie.

The application provides an interactive and user-friendly interface where users can enter movie details and instantly receive a predicted IMDb rating.

---

# ✨ Features

- 🎯 Predict IMDb movie ratings
- 📅 Select release year
- ⏱ Enter movie duration in hours and minutes
- 👍 Input IMDb votes
- 🎭 Choose movie genre
- 🎬 Select director
- ⭐ Select lead and supporting actors
- 📊 Display predicted IMDb rating
- 📈 Rating progress bar
- 📝 Movie summary
- 🌙 Professional dark-themed Streamlit interface

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

---

# 📂 Project Structure

```
Movie_Rating_Prediction/
│
├── app.py
├── movie_prediction.py
├── IMDb Movies India.csv
├── movie_rating_model.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── dashboard.png
    └── prediction.png
```

---

# 📊 Machine Learning Workflow

1. Load IMDb India Movies dataset
2. Clean and preprocess the data
3. Encode categorical features
4. Train Random Forest Regressor
5. Evaluate model performance
6. Save the trained model
7. Build Streamlit web application
8. Predict IMDb movie ratings

---

# 📈 Model Performance

| Metric | Value |
|---------|-------|
| MAE | 0.826 |
| RMSE | 1.095 |
| R² Score | 0.355 |

---

# 📷 Application Preview

## Dashboard

![Dashboard](screenshots/dashboard.png)

## Prediction Result

![Prediction](screenshots/prediction.png)

---

# 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Movie_Rating_Prediction.git
```

### Navigate to the project folder

```bash
cd Movie_Rating_Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 🎯 Internship Objectives Completed

- ✅ Dataset Loading
- ✅ Data Preprocessing
- ✅ Feature Encoding
- ✅ Machine Learning Model Training
- ✅ Model Evaluation
- ✅ Movie Rating Prediction
- ✅ Interactive Streamlit Web Application

---

# 👨‍💻 Author

**A. Santhosh**

**B.Tech – Computer Science and Engineering (Data Science)**

Machine Learning Internship Project

---

# 📄 License

This project is developed for educational and internship purposes.