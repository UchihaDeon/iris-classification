import pandas as pd

def load_data(path="data/iris.csv"):
    df = pd.read_csv(path)
    return df
def prepare_features_labels(df):
    # Separate features (X) and labels (y)
    X = df.drop(columns=["species"])
    y = df["species"]
    return X, y


def clean_data(df):
    df = df.drop_duplicates()
    return df