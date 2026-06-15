import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)
    le_activity = LabelEncoder()
    le_diet = LabelEncoder()
    df["Activity_Level"] = le_activity.fit_transform(df["Activity_Level"])
    df["Diet_Type"] = le_diet.fit_transform(df["Diet_Type"])
    return df, le_activity, le_diet
