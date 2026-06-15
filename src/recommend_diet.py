import pickle, pandas as pd

model = pickle.load(open("../models/diet_model.pkl","rb"))
activity_encoder = pickle.load(open("../models/activity_encoder.pkl","rb"))
diet_encoder = pickle.load(open("../models/diet_encoder.pkl","rb"))

age = int(input("Enter Age: "))
weight = float(input("Enter Weight (kg): "))
activity = input("Activity Level (Low/Medium/High): ")
diet = input("Diet Type (Veg/NonVeg): ")

data = pd.DataFrame([[age, weight,
activity_encoder.transform([activity])[0],
diet_encoder.transform([diet])[0]]],
columns=["Age","Weight","Activity_Level","Diet_Type"])

print("Recommended Daily Calories:", round(model.predict(data)[0]))
