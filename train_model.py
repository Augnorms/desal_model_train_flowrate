import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
# pyrefly: ignore [missing-import]
import joblib
import os

# Load data
df = pd.read_excel("clean_patterned_desalination_dataset.xlsx")

# Define features (everything except the two Permeate targets)
X = df.drop(columns=["Permeate Flowrate (m3/h)", "Permeate TDS (ppm)"])

# Define PERMEATE targets (the product water we want to predict)
y_permeate_flowrate = df["Permeate Flowrate (m3/h)"]  # Permeate flow in m³/h
y_permeate_tds = df["Permeate TDS (ppm)"]              # Permeate quality in ppm

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]} columns")
print(f"\n🎯 Predicting TWO Permeate (Product Water) targets:")
print(f"   1. Permeate Flowrate (m³/h) - How much product water we produce")
print(f"   2. Permeate TDS (ppm) - How clean the product water is")

# ==========================================
# Split into training and testing sets
# Use the SAME split for both targets
# ==========================================

X_train, X_test, y_train_permeate_flowrate, y_test_permeate_flowrate = train_test_split(
    X, y_permeate_flowrate, test_size=0.2, random_state=42
)

X_train_tds, X_test_tds, y_train_permeate_tds, y_test_permeate_tds = train_test_split(
    X, y_permeate_tds, test_size=0.2, random_state=42
)

print(f"\n✅ Training data: {X_train.shape[0]} rows")
print(f"✅ Testing data: {X_test.shape[0]} rows")

# ==========================================
# Train PERMEATE FLOWRATE Model
# ==========================================

print("\n🔧 Training Permeate Flowrate Model...")
model_permeate_flowrate = RandomForestRegressor(n_estimators=100, random_state=42)
model_permeate_flowrate.fit(X_train, y_train_permeate_flowrate)

# Evaluate Permeate Flowrate Model
y_pred_permeate_flowrate = model_permeate_flowrate.predict(X_test)
r2_permeate_flowrate = r2_score(y_test_permeate_flowrate, y_pred_permeate_flowrate)
mae_permeate_flowrate = mean_absolute_error(y_test_permeate_flowrate, y_pred_permeate_flowrate)

print(f"\n📈 PERMEATE FLOWRATE Model Performance:")
print(f"   R² Score: {r2_permeate_flowrate:.4f} (0-1, higher is better)")
print(f"   Mean Absolute Error: {mae_permeate_flowrate:.2f} m³/h")
print(f"   📊 Predicting: Permeate Flowrate (product water quantity)")

# ==========================================
# Train PERMEATE TDS Model
# ==========================================

print("\n🔧 Training Permeate TDS Model...")
model_permeate_tds = RandomForestRegressor(n_estimators=100, random_state=42)
model_permeate_tds.fit(X_train_tds, y_train_permeate_tds)

# Evaluate Permeate TDS Model
y_pred_permeate_tds = model_permeate_tds.predict(X_test_tds)
r2_permeate_tds = r2_score(y_test_permeate_tds, y_pred_permeate_tds)
mae_permeate_tds = mean_absolute_error(y_test_permeate_tds, y_pred_permeate_tds)

print(f"\n📈 PERMEATE TDS Model Performance:")
print(f"   R² Score: {r2_permeate_tds:.4f} (0-1, higher is better)")
print(f"   Mean Absolute Error: {mae_permeate_tds:.2f} ppm")
print(f"   📊 Predicting: Permeate TDS (product water quality)")

# ==========================================
# Save both models
# ==========================================

# Create directory if it doesn't exist
os.makedirs("models/saved_models", exist_ok=True)

joblib.dump(model_permeate_flowrate, "models/saved_models/permeate_flowrate_model_rf.pkl")
joblib.dump(model_permeate_tds, "models/saved_models/permeate_flowtds_model_rf.pkl")

print("\n💾 Both Permeate prediction models saved successfully!")
print(f"   - permeate_flowrate_model_rf.pkl  → Predicts PERMEATE Flowrate (m³/h)")
print(f"   - permeate_flowtds_model_rf.pkl   → Predicts PERMEATE TDS (ppm)")

# ==========================================
# Feature importance
# ==========================================

print("\n🔍 Top 5 Features for Predicting PERMEATE FLOWRATE:")
feature_importance_flow = pd.DataFrame({
    'feature': X.columns,
    'importance': model_permeate_flowrate.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_flow.head(5).to_string(index=False))

print("\n🔍 Top 5 Features for Predicting PERMEATE TDS:")
feature_importance_tds = pd.DataFrame({
    'feature': X.columns,
    'importance': model_permeate_tds.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_tds.head(5).to_string(index=False))

# ==========================================
# Summary of what we're predicting
# ==========================================

print("\n" + "="*70)
print("📋 SUMMARY: Permeate (Product Water) Prediction Models")
print("="*70)
print(f"🎯 Target 1: PERMEATE FLOWRATE (m³/h)")
print(f"   → How much product water the system produces")
print(f"   → Model: flowrate_model_rf.pkl")
print(f"   → Performance: R² = {r2_permeate_flowrate:.4f}, MAE = {mae_permeate_flowrate:.2f} m³/h")
print()
print(f"🎯 Target 2: PERMEATE TDS (ppm)")
print(f"   → How clean the product water is")
print(f"   → Model: flowtds_model_rf.pkl")
print(f"   → Performance: R² = {r2_permeate_tds:.4f}, MAE = {mae_permeate_tds:.2f} ppm")
print("="*70)