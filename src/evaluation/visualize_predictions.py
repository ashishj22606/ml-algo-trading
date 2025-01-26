import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_actual_vs_predicted_by_ticker(df):
    """
    Plot actual vs. predicted values for each ticker with different colors.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'Date', 'Ticker', 'Actual', and 'Predicted'.
    """
    unique_tickers = df['Ticker'].unique()
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    for ticker in unique_tickers:
        ticker_data = df[df['Ticker'] == ticker]
        plt.figure(figsize=(10, 6))

        # Plot actual values
        sns.lineplot(
            x=ticker_data['Date'], 
            y=ticker_data['Actual'], 
            label='Actual', 
            color='blue', 
            lw=2
        )
        
        # Plot predicted values
        sns.lineplot(
            x=ticker_data['Date'], 
            y=ticker_data['Predicted'], 
            label='Predicted', 
            color='orange', 
            lw=2, 
            linestyle='--'
        )

        # Add titles and labels
        plt.title(f"Actual vs. Predicted for {ticker}", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Stock Price", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.xticks(rotation=45)

        # Save or display
        plt.tight_layout()
        path = f"src/data/plots/{ticker}_actual_vs_predicted.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()
