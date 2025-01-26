from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from joblib import dump
import datetime
import os
import json

class RegressionModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the regression model.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the regression model.
        """
        return self.model.predict(X)
        
    def predict_future(self, df: pd.DataFrame, steps: int = 5, feature_names: list = None) -> pd.DataFrame:
        """
        Predict future prices iteratively for each ticker for the next `steps` business days.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the historical data and features.
            steps (int): Number of future steps to predict.
            feature_names (list): List of feature names used during training.

        Returns:
            pd.DataFrame: DataFrame containing future predictions for all tickers.
        """

        future_predictions = []

        # Initialize NYSE calendar and schedule
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date="2022-01-01", end_date="2026-12-31")

        # Ensure schedule.index is a timezone-naive pd.Timestamp
        schedule.index = schedule.index.tz_localize('UTC')

        # Group by Ticker
        tickers = df['Ticker'].unique()
        for ticker in tickers:
            # Filter data for the current ticker
            ticker_data = df[df['Ticker'] == ticker].copy()

            # Ensure 'Date' column is in pd.Timestamp format
            ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], utc=True)

            for step in range(steps):
                # Sort ticker data and get the latest row
                ticker_data = ticker_data.sort_values(by='Date')
                last_row = ticker_data.iloc[-1]

                # Get the current date as a pd.Timestamp
                current_date = pd.Timestamp(last_row['Date']).tz_convert('UTC')

                # Get the next valid trading day as a pd.Timestamp
                try:
                    next_date = schedule.index[schedule.index > current_date].min()
                except Exception as e:
                    print(f"Error: {e}")
                    print(f" schedule.index is {schedule.index} and current_date is {current_date}")
                    break

                # Generate features for the next day
                next_features = {
                    'Date': next_date,
                    'Ticker': ticker,
                    'Open': (last_row['Close'] + last_row['Open']) / 2,  # Smoothed Open
                    'High': last_row['Close'] * (1 + np.random.uniform(0.01, 0.03)),  # Simulate High
                    'Low': last_row['Close'] * (1 - np.random.uniform(0.01, 0.03)),   # Simulate Low
                    'Volume': last_row['Volume'] * (1 + np.random.uniform(-0.1, 0.1)),  # Simulate Volume
                    'MA_5': ticker_data['Close'].iloc[-5:].mean() if len(ticker_data) >= 5 else ticker_data['Close'].mean(),
                    'MA_20': ticker_data['Close'].iloc[-20:].mean() if len(ticker_data) >= 20 else ticker_data['Close'].mean(),
                    'Daily_Return': 0,  # Placeholder
                    'Volatility': last_row['Volatility'],  # Placeholder
                    'Ticker_Label': last_row['Ticker_Label'],  # Label encoding
                }

                # Convert features into a DataFrame
                next_features_df = pd.DataFrame([next_features])

                # Ensure feature alignment
                next_X = next_features_df.drop(columns=['Date', 'Ticker'])
                next_X = next_X[feature_names]  # Align features with training features

                # Predict the next day's price
                next_prediction = self.model.predict(next_X)[0]

                # Add prediction to the results
                next_features_df['Predicted_Close'] = next_prediction
                future_predictions.append(next_features_df)

                # Append the prediction to `ticker_data` for the next iteration
                next_features_df['Close'] = next_prediction  # Use predicted Close as actual Close
                ticker_data = pd.concat([ticker_data, next_features_df], ignore_index=True)

        # Combine all ticker predictions into a single DataFrame
        return pd.concat(future_predictions, ignore_index=True)

    def save_model(self, path: str):
        """
        Save the model and its metadata to a file.
        
        Parameters:
            path (str): The file path where the model will be saved.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        dump(self.model, path)

        # Create metadata
        metadata = {
            "model_name": "RandomForestRegressor",
            "train_date": datetime.datetime.now().isoformat(),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "train_data": "training_data.csv"
        }

        # Save metadata to a file in the same directory as the model
        metadata_path = os.path.join(os.path.dirname(path), f'{metadata["model_name"]}_model_metadata.json')
        if os.path.exists(metadata_path):
            # Load existing metadata
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        else:
            # Create a new list if no file exists
            existing_metadata = []

        # Append new metadata to the list
        existing_metadata.append(metadata)

        # Save the updated metadata back to the file
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=4)

        print(f"Metadata appended to {metadata_path}")