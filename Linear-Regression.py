import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Generate a realistic dataset (e.g., House Size in sqm vs Price in thousands)
np.random.seed(42)
X = np.random.rand(100, 1) * 200              # House sizes between 0 and 200 sqm
y = 3.5 * X.squeeze() + 40 + np.random.randn(100) * 15  # True relation + random market noise

# 2. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the unseen test set
y_pred = model.predict(X_test)

# 5. Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Parameters ---")
print(f"Slope (Price increase per sqm): {model.coef_[0]:.2f}")
print(f"Intercept (Base price): {model.intercept_:.2f}")

print(f"\n--- Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.2f} (Lower is better)")
print(f"R-squared Score (R2): {r2:.4f} (Closer to 1.0 means great fit)")