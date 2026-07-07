import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df = pd.read_csv('SeoulBikeData.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['DayOfWeek'] = df['Date'].dt.dayofweek      # 0=Monday, 6=Sunday
df['Month'] = df['Date'].dt.month              # 1-12
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)  # 1 if Sat/Sun, else 0

# Create time-of-day categories
df['IsRushHour'] = df['Hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df['IsNight'] = df['Hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

df['Seasons_Encoded'] = df['Seasons'].map({'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3})
df['Holiday_Encoded'] = df['Holiday'].map({'No Holiday': 0, 'Holiday': 1})
df['Functioning_Encoded'] = df['Functioning Day'].map({'No': 0, 'Yes': 1})


target = 'Rented Bike Count'

feature_cols = [
    'Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Dew point temperature(C)', 'Solar Radiation (MJ/m2)',
    'Rainfall(mm)', 'Snowfall (cm)',
    'DayOfWeek', 'Month', 'IsWeekend', 'IsRushHour', 'IsNight',
    'Seasons_Encoded', 'Holiday_Encoded', 'Functioning_Encoded'
]
df = df.sort_values('Date')

x = df[feature_cols]
y= df[target]

print("Features (x):", x.shape)
print("Target (y):", y.shape)
print("\nFeatures being used:")
print(x.columns.tolist())




split_index = int(len(df) * 0.8)
x_train = x.iloc[:split_index]
y_train = y.iloc[:split_index]
x_test = x.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training set size: {len(x_train)} samples")
print(f"Testing set size: {len(x_test)} samples")

print(f"Train data Range {df.iloc[0]['Date'].date()} to {df.iloc[split_index-1]['Date'].date()}")
print(f"Test data Range {df.iloc[split_index]['Date'].date()} to {df.iloc[-1]['Date'].date()}")



model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")


importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()