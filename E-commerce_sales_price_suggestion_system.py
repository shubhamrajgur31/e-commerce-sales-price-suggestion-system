# Product Recommendation System 
# Author: Shubham Rajguru

# Step 1: Import Libraries
import pandas as pd                     # For handling datasets
import numpy as np                      # For numerical operations
import matplotlib.pyplot as plt         # For graph visualization
from sklearn.linear_model import LinearRegression  # ML model
from sklearn.metrics import r2_score    # Model evaluation metric

# Step 2: Create Dataset
# Simulated dataset of Amazon product sales
data = {
    "Units_Sold": [150, 220, 330, 400, 550, 620, 710, 850, 920, 1050],
    "Avg_Review": [3.8, 4.0, 4.1, 4.3, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    "Current_Price": [18, 22, 25, 28, 30, 34, 37, 40, 42, 45],
    "Suggested_Price": [20, 24, 27, 31, 33, 36, 39, 43, 45, 48]
}

# Convert to DataFrame for easy manipulation
df = pd.DataFrame(data)
print("\n Dataset Preview:\n", df.head())

# Step 3: Features (X) & Target (y)
X = df[["Units_Sold", "Avg_Review", "Current_Price"]]  # Independent variables
y = df["Suggested_Price"]                              # Target variable

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)  # Fit model with dataset

# Predictions
y_pred = model.predict(X)

# Step 5: Evaluation
r2 = r2_score(y, y_pred)
print("\n Model Performance:")
print("R² Score:", round(r2, 4))
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Step 6: Matplotlib Graph
plt.figure(figsize=(8, 5))
plt.plot(y.values, label="Actual Price", marker="o")
plt.plot(y_pred, label="Predicted Price", marker="x")

plt.title("Amazon Sales Price Suggestion System")
plt.xlabel("Product Index")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: ASCII Graph (Terminal-Friendly)
print("\n ASCII Graph: Actual vs Predicted Price\n")

max_val = max(max(y), max(y_pred))
scale = 50 / max_val  # Scale prices to fit in graph width

for i in range(len(y)):
    actual_pos = int(y.iloc[i] * scale)
    pred_pos = int(y_pred[i] * scale)
    line = [" "] * 60
    line[actual_pos] = "A"   # Actual Price
    line[pred_pos] = "P"     # Predicted Price
    print(f"Product {i+1:2d} | {''.join(line)}")

print("\nLegend: A = Actual Price, P = Predicted Price")

# Step 8: Example Prediction
new_data = np.array([[1000, 4.7, 40]])  # [Units_Sold, Avg_Review, Current_Price]
predicted_price = model.predict(new_data)[0]

print("\n💡 Example Prediction:")
print("For Units_Sold=1000, Avg_Review=4.7, Current_Price=40 → Suggested Price =",
      round(predicted_price, 2))
