import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("advertising.csv")

# Select Features
X = df[["TV"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred)

print("="*40)
print("Model Trained Successfully")
print(f"R² Score : {accuracy:.2f}")
print("="*40)

# Save Model
joblib.dump(model, "sales_model.pkl")

print("Model saved as sales_model.pkl")