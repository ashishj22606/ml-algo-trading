import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators such as moving averages, RSI, etc., to the DataFrame.
    """
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
    df = add_bollinger_band_width(df)
    df.dropna(inplace=True)
    return df


def preprocess_all_data(filepaths: list) -> pd.DataFrame:
    """
    Preprocess all data files and combine them into a single DataFrame.
    """
    dfs = []
    for filepath in filepaths:
        ticker = os.path.basename(filepath).split('_')[1].split('.')[0]
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)

        # Add technical indicators
        df = add_technical_indicators(df)
        df['Ticker'] = ticker
        dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Encode Ticker column globally
    combined_df['Ticker'] = combined_df['Ticker'].astype('category')
    combined_df['Ticker_Label'] = combined_df['Ticker'].cat.codes

    # One-hot encode Ticker column
    ticker_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
    ticker_encoded = ticker_encoder.fit_transform(combined_df[['Ticker']])
    ticker_columns = [f'Ticker_{cat}' for cat in ticker_encoder.categories_[0][1:]]
    ticker_encoded_df = pd.DataFrame(ticker_encoded, columns=ticker_columns, index=combined_df.index)

    # Combine one-hot encoded columns with the DataFrame
    combined_df = pd.concat([combined_df, ticker_encoded_df], axis=1)

    return combined_df


def add_bollinger_band_width(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Bollinger Band width as a volatility measure to the DataFrame.
    """
    # Moving average and standard deviation
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['STD_DEV'] = df['Close'].rolling(window=window).std()
    
    # Bollinger Bands
    df['Upper_Band'] = df['MA'] + (2 * df['STD_DEV'])
    df['Lower_Band'] = df['MA'] - (2 * df['STD_DEV'])
    
    # Bollinger Band Width
    df['Volatility'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA']
    # df['Volatility'] = df.groupby('Ticker')['Volatility'].transform(lambda x: (x - x.mean()) / x.std())

    return df

def train_test_split_by_ticker(df, train_ratio=0.8):
    """
    Perform train-test split per ticker while preserving chronological order.
    """
    train_frames = []
    test_frames = []

    for ticker, group in df.groupby('Ticker', observed=False):
        # Sort by Date to ensure chronological order
        group = group.sort_values(by='Date')

        # Calculate split index
        train_size = int(len(group) * train_ratio)

        # Split the group
        train_frames.append(group.iloc[:train_size])
        test_frames.append(group.iloc[train_size:])

    # Concatenate all train and test splits
    train_data = pd.concat(train_frames)
    test_data = pd.concat(test_frames)

    return train_data, test_data
