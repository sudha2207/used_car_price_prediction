import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("data/cardekho_dataset.csv")

# 2. CLEAN DATA
# remove unwanted columns
df.drop(["Unnamed: 0", "car_name", "model"], axis=1, inplace=True)

# clean mileage
df["mileage"] = df["mileage"].astype(str).str.replace(" kmpl", "")
df["mileage"] = pd.to_numeric(df["mileage"], errors='coerce')

# clean engine
df["engine"] = df["engine"].astype(str).str.replace(" CC", "")
df["engine"] = pd.to_numeric(df["engine"], errors='coerce')

# clean max_power
df["max_power"] = df["max_power"].astype(str).str.replace(" bhp", "")
df["max_power"] = pd.to_numeric(df["max_power"], errors='coerce')

# remove missing values (if any)
df.dropna(inplace=True)

# 3. ENCODE DATA
df = pd.get_dummies(df, drop_first=True)

# 4. SPLIT DATA
from sklearn.model_selection import train_test_split

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TRAIN MODEL
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. PREDICT
predictions = model.predict(X_test)

print("Sample Predictions:")
print(predictions[:5])

# 7. EVALUATE MODEL
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)
print("\nR2 Score:", score)

# 8. FEATURE IMPORTANCE
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns)
filtered_features = feature_importance[~feature_importance.index.str.contains("brand_")]
top_features = filtered_features.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh')

plt.title("Top Important Features Affecting Car Price")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.xlim(0, 0.15)

plt.tight_layout()
plt.show()