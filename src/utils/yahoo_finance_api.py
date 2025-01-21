import yfinance as yf
from typing import List, Dict, Any
import pandas as pd

class YFinanceUtils:
    @staticmethod
    def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker symbol.

        :param ticker: Stock ticker symbol (e.g., "AAPL" for Apple).
        :param start_date: Start date in the format "YYYY-MM-DD".
        :param end_date: End date in the format "YYYY-MM-DD".
        :return: DataFrame containing historical stock data.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_crypto_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data using yfinance.

        :param ticker: Cryptocurrency ticker symbol (e.g., "BTC-USD" for Bitcoin).
        :param start_date: Start date in the format "YYYY-MM-DD".
        :param end_date: End date in the format "YYYY-MM-DD".
        :return: DataFrame containing historical cryptocurrency data.
        """
        try:
            crypto = yf.Ticker(ticker)
            data = crypto.history(start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_multiple_data(tickers: List[str], start_date: str, end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers (stocks or cryptocurrencies).

        :param tickers: List of ticker symbols.
        :param start_date: Start date in the format "YYYY-MM-DD".
        :param end_date: End date in the format "YYYY-MM-DD".
        :return: Dictionary where keys are ticker symbols and values are DataFrames with historical data.
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = YFinanceUtils.fetch_stock_data(ticker, start_date, end_date, interval)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results

    @staticmethod
    def get_ticker_info(ticker: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific ticker (stock or crypto).

        :param ticker: Ticker symbol.
        :return: Dictionary containing detailed information about the ticker.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}

    @staticmethod
    def get_live_price(ticker: str) -> float:
        """
        Fetch the live price for a given ticker.

        :param ticker: Ticker symbol.
        :return: Live price of the ticker.
        """
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")["Close"].iloc[-1]
            return price
        except Exception as e:
            print(f"Error fetching live price for {ticker}: {e}")
            return 0.0

