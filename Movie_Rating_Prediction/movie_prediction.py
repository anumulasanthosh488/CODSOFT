import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("Dataset Loaded Successfully!")
print(df.shape)

# Remove missing ratings
df = df.dropna(subset=["Rating"])

# Fill missing values
df.fillna("Unknown", inplace=True)

# Remove unwanted symbols
df["Year"] = df["Year"].astype(str).str.extract("(\d+)")[0]
df["Duration"] = df["Duration"].astype(str).str.extract("(\d+)")[0]

# Convert to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0)

# Convert Votes
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)

# Encode categorical columns
encoders = {}

for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, "label_encoders.pkl")
# Features
X = df[["Year","Duration","Votes","Genre","Director","Actor 1","Actor 2","Actor 3"]]

# Target
y = df["Rating"]

# Train Test Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

# Model
model=RandomForestRegressor(
    n_estimators=100,
    random_state=42)

model.fit(X_train,y_train)

# Prediction
y_pred=model.predict(X_test)

print("\nModel Performance")
print("----------------------")
print("MAE :",mean_absolute_error(y_test,y_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score :",r2_score(y_test,y_pred))

# Predict a movie rating

sample_movie = [[2020, 120, 5000, 10, 20, 30, 40, 50]]

prediction = model.predict(sample_movie)

print("\nPredicted Movie Rating:", round(prediction[0], 2))



joblib.dump(model, "movie_rating_model.pkl")

print("Model saved successfully!")