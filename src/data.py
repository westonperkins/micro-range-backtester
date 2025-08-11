import os
import pandas as pd

def load_data_dir(data_dir: str) -> dict:
    """
    Load all CSVs in data_dir. Each CSV must contain:
    date (YYYY-MM-DD), open, high, low, close, volume, ticker
    Returns dict[ticker] -> pd.DataFrame sorted by date asc with a DatetimeIndex.
    """
    frames = {}
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skipping {fn}: {e}")
            continue
        required = {"date","open","high","low","close","volume","ticker"}
        if not required.issubset(set(df.columns)):
            print(f"Skipping {fn}: missing required columns {required - set(df.columns)}")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")
        ticker = str(df["ticker"].iloc[0])
        frames[ticker] = df
    return frames
