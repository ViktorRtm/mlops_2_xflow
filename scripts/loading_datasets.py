import pandas as pd


def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)

def upload_dataset(df: pd.DataFrame, dataset_path: str) -> pd.DataFrame:
    return df.to_csv(dataset_path, index=False)