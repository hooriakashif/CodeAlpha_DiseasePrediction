import os
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

CSV_URL = "https://raw.githubusercontent.com/anshuldutt9/Heart-Disease-UCI-ML/master/heart.csv"

def download_dataset(url=CSV_URL, dest=CSV_PATH):
    if os.path.exists(dest):
        print(f"Dataset already exists at {dest}")
        return dest
    try:
        print("Downloading dataset...")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"Saved dataset to {dest}")
        return dest
    except Exception as e:
        print("Download failed. Please manually place heart.csv inside data folder.")
        print("Error:", e)
        return None

def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        print("Dataset not found. Attempting to download it.")
        download_dataset()
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset still not found. Please place heart.csv in the data folder.")
    df = pd.read_csv(path)
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print(df.head())
    return df

if __name__ == "__main__":
    load_data()
