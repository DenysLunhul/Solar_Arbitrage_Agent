import pandas as pd

def drop_features(df):
    DROP_FEATURES = [
        "timestamp",
        "Month", "Day", "Hour", "Day_of_week", "Minute",
        "DAM_Vol_Sale", "DAM_Vol_Buy"
    ]

    df.drop(columns=[DROP_FEATURES], inplace=True)
    return df


#TODO
def normalize_features(df):
    return df