import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000
dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

marketing_spend = np.random.randint(1000, 5000, n)
discount = np.random.uniform(0, 30, n)
foot_traffic = np.random.randint(100, 1000, n)
competitor_price = np.random.uniform(10, 50, n)

# Day of week and month effects
day_of_week = pd.Series(dates).dt.dayofweek.values
month = pd.Series(dates).dt.month.values
weekend_boost = np.where(day_of_week >= 5, 1.3, 1.0)
seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)

# Generate sales with realistic relationships
sales = (
    200
    + 0.05 * marketing_spend
    + 8 * discount
    + 0.4 * foot_traffic
    - 5 * competitor_price
) * weekend_boost * seasonal_factor

# Add noise
sales += np.random.normal(0, 50, n)
sales = np.clip(sales, 0, None).round(2)

data = pd.DataFrame({
    "date": dates,
    "marketing_spend": marketing_spend,
    "discount": discount.round(2),
    "foot_traffic": foot_traffic,
    "competitor_price": competitor_price.round(2),
    "day_of_week": day_of_week,
    "month": month,
    "sales": sales,
})

data.to_csv("sales_data.csv", index=False)
print(f"Generated {len(data)} rows -> sales_data.csv")
print(data.describe())
