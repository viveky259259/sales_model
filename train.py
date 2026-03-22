import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("sales_data.csv")

# Features and target
feature_cols = ["marketing_spend", "discount", "foot_traffic", "competitor_price", "day_of_week", "month"]
X = df[feature_cols]
y = df["sales"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# Feature importance
print("\nFeature Importance:")
for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s} {imp:.4f}")

# Save model
joblib.dump(model, "sales_model.pkl")
print("\nModel saved to sales_model.pkl")
