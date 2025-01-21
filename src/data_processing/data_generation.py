import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.yahoo_finance_api import YFinanceUtils

def load_tickers(file_path: str) -> dict:
    """
    Load tickers from a JSON file.
    
    :param file_path: Path to the JSON file containing tickers.
    :return: Dictionary with stock and crypto tickers.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading tickers from {file_path}: {e}")
        return {"stocks": [], "cryptos": []}

def save_to_csv(data: dict, folder_path: str, asset_type: str):
    """
    Save fetched data to CSV files.
    
    :param data: Dictionary with tickers as keys and DataFrames as values.
    :param folder_path: Folder path where CSV files will be saved.
    :param asset_type: Type of asset ('stock' or 'crypto').
    """
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    for ticker, df in data.items():
        if not df.empty:
            file_path = os.path.join(folder_path, f"{asset_type}_{ticker}.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
        else:
            print(f"No data available for {ticker}. Skipping CSV generation.")

def data_generation():
    # Load tickers from the JSON file
    tickers = load_tickers("src/data/tickers.json")
    
    # Fetch and save data for stocks
    print("Fetching stock data:")
    stock_data = YFinanceUtils.fetch_multiple_data(tickers["stocks"], "2022-01-01", "2022-12-31", interval="1d")
    save_to_csv(stock_data, "src/data/stock", "stock")

    # Fetch and save data for cryptocurrencies
    print("Fetching cryptocurrency data:")
    crypto_data = YFinanceUtils.fetch_multiple_data(tickers["cryptos"], "2022-01-01", "2022-12-31", interval="1d")
    save_to_csv(crypto_data, "src/data/crypto", "crypto")

if __name__ == "__main__":
    data_generation()
