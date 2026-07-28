import pandas as pd
from pandas.core.common import random_state 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

#load data
df = pd.read_excel("clean_patterned_desalination_dataset.xlsx")

#[Feed Flowrate (m3/h), Feed TDS (ppm), Feed Salinity (g/L), Feed Temperature (掳C), Permeate Flowrate (m3/h), Permeate TDS (ppm),]
#[Recovery (%), Flux (LMH), Pressure Drop (bar)]

#print(df.head())
#print(f"length of row {len(df)}")
#print(f"length of column {len(df.columns)}")
#print(df.columns.to_list())

#print(df.isnull().sum()) 

# define the target and features
features = df.drop(columns=["Permeate Flowrate (m3/h)", "Permeate TDS (ppm)"])
target_flowrate = df["Permeate Flowrate (m3/h)"]
target_tds = df["Permeate TDS (ppm)"]

X = features
y_flowrate = target_flowrate
y_tds = target_tds

# print(f"X shape {X.shape}")
# print(f"y flowrate {y_flowrate.shape}")
# print(f"y tds {y_tds.shape}")


# ==========================================
# NEW: Split into training and testing sets
# ==========================================

X_train, X_test, y_flowrate_train, y_flowrate_test = train_test_split(X, y_flowrate, test_size=0.2, random_state=42)

# print(f"✅ Training data: {X_train.shape[0]} rows")
# print(f"✅ Testing data: {X_test.shape[0]} rows")

# ==========================================
# Create and train model
# ==========================================

model_flowrate = RandomForestRegressor(n_estimators=100, random_state=42)
#model_flowrate = LinearRegression()
model_flowrate.fit(X_train, y_flowrate_train)

print("model flowrate trained successfully !!")

# ==========================================
# model prediction
# ==========================================

y_flowrate_pred = model_flowrate.predict(X_test)

#model evaluation
r2 = r2_score(y_flowrate_test, y_flowrate_pred)
mae = mean_absolute_error(y_flowrate_test, y_flowrate_pred)

print(f"\n📈 Model 1 Performance:")
print(f"   R² Score: {r2:.4f} (0-1, higher is better)")
print(f"   Mean Absolute Error: {mae:.2f} m³/h")


# Save the model after training
joblib.dump(model_flowrate, "models/saved_models/flowrate_model_rf.pkl")

print("\n💾 Random Forest model saved successfully!")

# print("\n🔍 Coefficients (how each feature affects flowrate):")
# for name, coef in zip(X.columns, model_flowrate.coef_):
#     print(f"   {name}: {coef:.4f}")