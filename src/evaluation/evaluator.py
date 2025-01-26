import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_name: str, metrics: dict = None):
        """
        Initialize the ModelEvaluator class.

        Parameters:
            model_name (str): Name of the model being evaluated.
            metrics (dict): A dictionary of metric names and their corresponding functions.
        """
        self.model_name = model_name
        self.metrics = metrics or {
            "MAE": self.mean_absolute_error,
            "RMSE": self.root_mean_squared_error,
            "R2": self.r_squared,
            "Directional Accuracy": self.directional_accuracy
        }
        self.results = {}

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(ModelEvaluator.mean_squared_error(y_true, y_pred))

    @staticmethod
    def r_squared(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        if len(y_true) <= 1 or len(y_pred) <= 1:
            return None  # Directional accuracy is not meaningful for 1 or fewer predictions
        correct_directions = np.sum(
            np.sign(np.array(y_true[1:]) - np.array(y_true[:-1])) ==
            np.sign(np.array(y_pred[1:]) - np.array(y_pred[:-1]))
        )
        return (correct_directions / (len(y_true) - 1)) * 100

    def evaluate_by_ticker(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, tickers: pd.Series):
        """
        Evaluate metrics grouped by ticker.

        Parameters:
            y_true (pd.Series): Actual target values.
            y_pred (pd.Series): Predicted target values.
            tickers (pd.Series): Corresponding tickers for the data.

        Returns:
            pd.DataFrame: Evaluation metrics for each ticker.
        """
        results = []
        grouped = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "Ticker": tickers}).groupby("Ticker", observed=False)
        
        for ticker, group in grouped:
            ticker_results = {
                "Ticker": ticker
            }
            for metric_name, metric_func in self.metrics.items():
                ticker_results[metric_name] = metric_func(group["y_true"].values, group["y_pred"].values)
            results.append(ticker_results)

        return pd.DataFrame(results)

    def print_results_by_ticker(self, results: pd.DataFrame):
        """
        Print evaluation results grouped by ticker.

        Parameters:
            results (pd.DataFrame): Evaluation metrics grouped by ticker.
        """
        print(f"\nEvaluation Results for {self.model_name} by Ticker:")
        print(results.to_string(index=False))

    def save_results_by_ticker_json(self, results: pd.DataFrame, save_path: str):
        """
        Save per-ticker evaluation results into a JSON file, appending to existing data.

        Parameters:
            results (pd.DataFrame): DataFrame of evaluation results by ticker.
            save_path (str): Path to the JSON file.
        """
        # Create the JSON structure for this run
        current_run = {
            "model_name": self.model_name,
            "evaluation_time": datetime.now().isoformat(),
            "results": results.to_dict(orient="records")
        }

        # Load existing data if the file exists
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = []

        # Append current results to the existing data
        all_results.append(current_run)

        # Save updated results back to the file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"Evaluation results appended to {save_path}")