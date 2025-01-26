import sys
from dotenv import load_dotenv
from src.data_processing.data_generation import data_generation
from src.data_processing.feature_engineering import preprocess_all_data, train_test_split_by_ticker
from src.models.regression_model import RegressionModel
import glob
import pandas as pd
import os
from src.evaluation.evaluator import ModelEvaluator

load_dotenv()

def main():
    
    data_generation()
    
    # Load and preprocess stock and crypto data
    stock_files = glob.glob("src/data/stock/stock_*.csv")
    crypto_files = glob.glob("src/data/crypto/crypto_*.csv")
    all_files = stock_files + crypto_files

    # Preprocess all data
    df = preprocess_all_data(all_files)

    # Prepare features and target
    X = df[['Date', 'Ticker', 'Ticker_Label', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'Daily_Return', 'Volatility', 'Close']]

    # Perform train-test split per ticker
    train_data, test_data = train_test_split_by_ticker(X, train_ratio=0.95)

    # Separate features and target
    X_train = train_data.drop(columns=['Date', 'Ticker', 'Close'])
    y_train = train_data['Close']
    X_test = test_data.drop(columns=['Date', 'Ticker', 'Close'])
    y_test = test_data['Close']

    # Extract feature names
    feature_names = X_train.columns.tolist()

    # Initialize and train regression model
    reg_model = RegressionModel(n_estimators=100)
    reg_model.train(X_train, y_train)
    
    # Save the trained model and its metadata
    model_save_path = "src/models/generated_models/random_forest_model.pkl"
    reg_model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate
    X_test = X_test[feature_names]  # Align test features
    predictions = reg_model.predict(X_test)
    
    evaluator = ModelEvaluator(model_name="RandomForestRegressor")
    evaluation_results_ticker = evaluator.evaluate_by_ticker(y_test, predictions, test_data["Ticker"])
    path = "src/data/RandomForestRegressor/random_forest_evaluation_results_by_ticker.json"
    evaluator.save_results_by_ticker_json(evaluation_results_ticker, path)

    # Combine predictions with actual values and metadata
    test_set_predictions = test_data[['Date', 'Ticker']].copy()
    test_set_predictions['Actual'] = y_test
    test_set_predictions['Predicted'] = predictions

    # Sort results by Ticker and Date (descending)
    test_set_predictions = test_set_predictions.sort_values(by=['Ticker', 'Date'], ascending=[True, False])

    # Save predictions to CSV
    path = "src/data/RandomForestRegressor/test_set_predictions_with_dates.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    test_set_predictions.to_csv(path, index=False)
    print("Test Set Predictions saved to test_set_predictions_with_dates.csv")
    
    # Predict future prices
    future_predictions = reg_model.predict_future(df, steps=5, feature_names=feature_names)
    
    # Save future predictions to CSV
    path = "src/data/RandomForestRegressor/future_predictions.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    future_predictions.to_csv(path, index=False)
    print("Future predictions saved to future_predictions.csv")

if __name__ == "__main__":
    main()