import os, pickle
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_and_preprocess

df, le_activity, le_diet = load_and_preprocess("../dataset/diet_data.csv")
X = df[["Age","Weight","Activity_Level","Diet_Type"]]
y = df["Calories"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("../models", exist_ok=True)

pickle.dump(model, open("../models/diet_model.pkl","wb"))
pickle.dump(le_activity, open("../models/activity_encoder.pkl","wb"))
pickle.dump(le_diet, open("../models/diet_encoder.pkl","wb"))

print("Model trained successfully!")
