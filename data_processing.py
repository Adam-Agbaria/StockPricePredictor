"""
Data loading and preprocessing for Stock Price Predictor
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_CONFIG, RANDOM_SEED

# Set random seed
np.random.seed(RANDOM_SEED)

def load_raw_data(csv_path):
    """
    Load raw CSV data and perform basic parsing.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        pandas DataFrame with parsed datetime index
    """
    df = pd.read_csv(csv_path, sep=";", header=None,
                     names=["date", "time", "open", "high", "low", "close", "volume"])

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                     format="%d/%m/%Y %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    
    # Convert to numeric and clean
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    
    return df

def add_technical_features(df):
    """
    Add technical analysis features to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional technical features
    """
    df = df.copy()
    
    # High-Low range as percentage of low
    df["hl_range"] = (df["high"] - df["low"]) / df["low"]
    
    # Moving averages
    df["close_ma"] = df["close"].rolling(20).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    

    
    return df

def create_target_variable(df, prediction_steps=1):
    """
    Create the target variable for prediction.
    
    Args:
        df: DataFrame with price data
        prediction_steps: Number of steps ahead to predict
    
    Returns:
        DataFrame with target variable added
    """
    df = df.copy()
    
    # Create target: future price (N steps ahead)
    df["future_price"] = df["close"].shift(-prediction_steps)
    
    return df

def build_sequences(features, targets, sequence_length):
    """
    Build sequences for LSTM training.
    
    Args:
        features: Feature array
        targets: Target array
        sequence_length: Length of input sequences
    
    Returns:
        X, y arrays for training
    """
    X, y = [], []
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)

def split_data(X, y, test_ratio=0.2):
    """
    Split data into train and test sets.
    
    Args:
        X: Feature sequences
        y: Target values
        test_ratio: Ratio for test set
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train, y_test):
    """
    Scale features and targets using MinMaxScaler.
    
    Args:
        X_train, X_test: Feature sequences
        y_train, y_test: Target values
    
    Returns:
        Scaled data and fitted scalers
    """
    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
    
    # Scale targets (prices)
    price_scaler = MinMaxScaler()
    y_train_scaled = price_scaler.fit_transform(y_train)
    y_test_scaled = price_scaler.transform(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            price_scaler, feature_scaler)

def load_and_preprocess(csv_path=None, sequence_length=None, test_ratio=None, 
                       prediction_steps=None):
    """
    Complete data loading and preprocessing pipeline.
    
    Args:
        csv_path: Path to CSV file (defaults to config)
        sequence_length: Length of input sequences (defaults to config)
        test_ratio: Test set ratio (defaults to config)
        prediction_steps: Steps ahead to predict (defaults to config)
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
                 price_scaler, feature_scaler)
    """
    # Use config defaults if not provided
    if csv_path is None:
        csv_path = DATA_CONFIG['csv_path']
    if sequence_length is None:
        sequence_length = DATA_CONFIG['sequence_length']
    if test_ratio is None:
        test_ratio = DATA_CONFIG['test_ratio']
    if prediction_steps is None:
        prediction_steps = DATA_CONFIG['prediction_steps']
    
    print(f"Loading data from: {csv_path}")
    print(f"Sequence length: {sequence_length}")
    print(f"Prediction steps: {prediction_steps}")
    print(f"Test ratio: {test_ratio:.1%}")
    
    # Load raw data
    df = load_raw_data(csv_path)
    print(f"Loaded {len(df):,} raw data points")
    
    # Add technical features
    df = add_technical_features(df)
    
    # Create target variable
    df = create_target_variable(df, prediction_steps)
    
    # Drop NaN values created by feature engineering and target creation
    df.dropna(inplace=True)
    print(f"After preprocessing: {len(df):,} data points")
    
    # Extract features and targets
    feature_columns = DATA_CONFIG['features']
    features = df[feature_columns].values
    targets = df["future_price"].values.reshape(-1, 1)
    
    print(f"Features used: {feature_columns}")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Build sequences
    X, y = build_sequences(features, targets, sequence_length)
    print(f"Sequence shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Scale data
    (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
     price_scaler, feature_scaler) = scale_data(X_train, X_test, y_train, y_test)
    
    print("Data preprocessing completed successfully!\n")
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            price_scaler, feature_scaler)

def get_data_info(csv_path=None):
    """
    Get basic information about the dataset.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary with dataset information
    """
    if csv_path is None:
        csv_path = DATA_CONFIG['csv_path']
    
    df = load_raw_data(csv_path)
    df = add_technical_features(df)
    df.dropna(inplace=True)
    
    info = {
        'total_records': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'price_range': (df['close'].min(), df['close'].max()),
        'volume_stats': df['volume'].describe(),
        'missing_data': df.isnull().sum().sum()
    }
    
    return info 