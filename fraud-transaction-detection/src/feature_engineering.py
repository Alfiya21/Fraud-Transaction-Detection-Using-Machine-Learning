
import pandas as pd

def create_features(df):
    df = df.copy()

    # Time-based features
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAYOFWEEK"] = df["TX_DATETIME"].dt.dayofweek
    df["IS_WEEKEND"] = df["TX_DAYOFWEEK"].isin([5, 6]).astype(int)

    # Customer behavior features
    df["CUST_AVG_AMOUNT"] = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform("mean")
    df["CUST_TX_COUNT"] = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform("count")
    df["AMOUNT_DEVIATION"] = df["TX_AMOUNT"] - df["CUST_AVG_AMOUNT"]

    # Terminal risk features
    df["TERMINAL_TX_COUNT"] = df.groupby("TERMINAL_ID")["TX_AMOUNT"].transform("count")
    df["TERMINAL_FRAUD_RATE"] = (
        df.groupby("TERMINAL_ID")["TX_FRAUD"].mean()
    ).reindex(df["TERMINAL_ID"]).values

    return df
