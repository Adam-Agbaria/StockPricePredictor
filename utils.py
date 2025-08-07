"""
Utility functions for Stock Price Predictor
"""

import numpy as np
import pandas as pd

from config import DATA_CONFIG, HORIZON_ANALYSIS
from data_processing import load_raw_data, add_technical_features

def analyze_prediction_horizons(csv_path=None, horizons=None):
    """
    Analyze different prediction horizons to help choose optimal one.
    Shows how average price movement changes with prediction distance.
    
    Args:
        csv_path: Path to CSV file (defaults to config)
        horizons: List of horizons to analyze (defaults to config)
    
    Returns:
        Dictionary with analysis results for each horizon
    """
    if csv_path is None:
        csv_path = DATA_CONFIG['csv_path']
    if horizons is None:
        horizons = HORIZON_ANALYSIS['horizons']
    
    print("PREDICTION HORIZON ANALYSIS")
    print("   Analyzing how price movement magnitude changes with prediction distance...")
    print("   Larger movements = better anti-zero-collapse performance\n")
    
    results = {}
    
    try:
        # Load and preprocess data exactly like main function
        df = load_raw_data(csv_path)
        df = add_technical_features(df)
        df.dropna(inplace=True)
        
        print(f"   Loaded {len(df):,} data points for analysis")
        
        for horizon in horizons:
            # Calculate delta for this horizon
            delta = df["close"].shift(-horizon) - df["close"]
            delta_clean = delta.dropna()
            
            if len(delta_clean) == 0:
                print(f"   {horizon:3d} steps ({horizon:3d}min): No valid data")
                results[horizon] = {
                    'avg_abs_delta': 0,
                    'std_delta': 0,
                    'zero_like_percentage': 100,
                    'valid_data_points': 0
                }
                continue
                
            avg_abs_delta = np.mean(np.abs(delta_clean))
            std_delta = np.std(delta_clean)
            
            # Safe calculation for zero-like percentage
            if len(delta_clean) > 0:
                zero_like = np.sum(np.abs(delta_clean) < 0.01) / len(delta_clean) * 100
            else:
                zero_like = 0.0
            
            time_mins = horizon * 1  # 1-minute data
            
            print(f"   {horizon:3d} steps ({time_mins:3d}min): avg|Δ|={avg_abs_delta:.4f}, std={std_delta:.4f}, ~zero={zero_like:.1f}%")
            
            results[horizon] = {
                'avg_abs_delta': avg_abs_delta,
                'std_delta': std_delta,
                'zero_like_percentage': zero_like,
                'valid_data_points': len(delta_clean),
                'time_minutes': time_mins
            }
        
        print("\nRecommendation:")
        print("   • Choose horizon where avg|Δ| > 0.05 and ~zero < 50%")
        print("   • Balance: longer horizon = better anti-collapse but harder to predict")
        print("   • For 1m data: 30-60 steps usually optimal\n")
        
        # Find recommended horizons based on criteria
        recommended_horizons = []
        for horizon, data in results.items():
            if (data['avg_abs_delta'] > HORIZON_ANALYSIS['min_avg_delta'] and 
                data['zero_like_percentage'] < HORIZON_ANALYSIS['max_zero_percentage']):
                recommended_horizons.append(horizon)
        
        if recommended_horizons:
            print(f"   Recommended horizons based on criteria: {recommended_horizons}")
        else:
            print("   No horizons meet the recommended criteria. Consider using longer horizons.")
        
        results['recommended'] = recommended_horizons
        
    except Exception as e:
        print(f"   WARNING: Error in horizon analysis: {e}")
        print("   Skipping horizon analysis, proceeding with training...\n")
        return {}
    
    return results

def get_optimal_horizon(csv_path=None, min_avg_delta=None, max_zero_percentage=None):
    """
    Get the optimal prediction horizon based on analysis criteria.
    
    Args:
        csv_path: Path to CSV file
        min_avg_delta: Minimum average absolute delta
        max_zero_percentage: Maximum percentage of near-zero values
    
    Returns:
        Optimal horizon (int) or None if no horizon meets criteria
    """
    if min_avg_delta is None:
        min_avg_delta = HORIZON_ANALYSIS['min_avg_delta']
    if max_zero_percentage is None:
        max_zero_percentage = HORIZON_ANALYSIS['max_zero_percentage']
    
    results = analyze_prediction_horizons(csv_path, HORIZON_ANALYSIS['horizons'])
    
    if not results or 'recommended' not in results:
        return None
    
    recommended = results['recommended']
    
    if not recommended:
        return None
    
    # Choose the shortest recommended horizon (more predictable)
    return min(recommended)

def calculate_data_statistics(csv_path=None):
    """
    Calculate comprehensive statistics about the dataset.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary with dataset statistics
    """
    if csv_path is None:
        csv_path = DATA_CONFIG['csv_path']
    
    print("DATASET STATISTICS")
    print("=" * 50)
    
    # Load raw data
    df = load_raw_data(csv_path)
    print(f"Raw data points: {len(df):,}")
    
    # Add technical features
    df_processed = add_technical_features(df)
    df_processed.dropna(inplace=True)
    print(f"After preprocessing: {len(df_processed):,}")
    
    # Date range
    date_range = df_processed.index.max() - df_processed.index.min()
    print(f"Date range: {df_processed.index.min()} to {df_processed.index.max()}")
    print(f"Total time span: {date_range}")
    
    # Price statistics
    price_stats = df_processed['close'].describe()
    print(f"\nPrice Statistics:")
    print(f"   Min price: {price_stats['min']:.2f}")
    print(f"   Max price: {price_stats['max']:.2f}")
    print(f"   Mean price: {price_stats['mean']:.2f}")
    print(f"   Std price: {price_stats['std']:.2f}")
    
    # Volume statistics
    vol_stats = df_processed['volume'].describe()
    print(f"\nVolume Statistics:")
    print(f"   Min volume: {vol_stats['min']:,.0f}")
    print(f"   Max volume: {vol_stats['max']:,.0f}")
    print(f"   Mean volume: {vol_stats['mean']:,.0f}")
    
    # Price volatility
    price_changes = df_processed['close'].pct_change().dropna()
    print(f"\nPrice Volatility:")
    print(f"   Daily return mean: {price_changes.mean():.4f}")
    print(f"   Daily return std: {price_changes.std():.4f}")
    print(f"   Max daily gain: {price_changes.max():.4f}")
    print(f"   Max daily loss: {price_changes.min():.4f}")
    
    # Missing data
    missing_data = df_processed.isnull().sum()
    print(f"\nMissing Data:")
    if missing_data.sum() == 0:
        print("   No missing data found")
    else:
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"   {col}: {missing} missing values")
    
    print("=" * 50)
    
    return {
        'raw_data_points': len(df),
        'processed_data_points': len(df_processed),
        'date_range': (df_processed.index.min(), df_processed.index.max()),
        'time_span': date_range,
        'price_stats': price_stats.to_dict(),
        'volume_stats': vol_stats.to_dict(),
        'price_volatility': {
            'mean_return': price_changes.mean(),
            'std_return': price_changes.std(),
            'max_gain': price_changes.max(),
            'max_loss': price_changes.min()
        },
        'missing_data': missing_data.to_dict()
    }

def validate_data_quality(csv_path=None, min_data_points=10000, max_missing_percentage=5.0):
    """
    Validate data quality for training.
    
    Args:
        csv_path: Path to CSV file
        min_data_points: Minimum required data points
        max_missing_percentage: Maximum acceptable missing data percentage
    
    Returns:
        Tuple of (is_valid, issues_list)
    """
    if csv_path is None:
        csv_path = DATA_CONFIG['csv_path']
    
    issues = []
    
    try:
        # Load and check data
        df = load_raw_data(csv_path)
        df_processed = add_technical_features(df)
        
        # Check data volume
        if len(df_processed) < min_data_points:
            issues.append(f"Insufficient data: {len(df_processed)} < {min_data_points} required")
        
        # Check missing data
        missing_percentage = (df_processed.isnull().sum().sum() / (len(df_processed) * len(df_processed.columns))) * 100
        if missing_percentage > max_missing_percentage:
            issues.append(f"Too much missing data: {missing_percentage:.1f}% > {max_missing_percentage}% allowed")
        
        # Check for constant values
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            if df_processed[col].nunique() <= 1:
                issues.append(f"Column '{col}' has constant values")
        
        # Check for extreme outliers
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df_processed[col].quantile(0.25)
            q3 = df_processed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            outlier_percentage = (outliers / len(df_processed)) * 100
            
            if outlier_percentage > 10:  # More than 10% outliers
                issues.append(f"Column '{col}' has {outlier_percentage:.1f}% extreme outliers")
        
        # Check date consistency
        if not df_processed.index.is_monotonic_increasing:
            issues.append("Data is not in chronological order")
        
        # Check for duplicate timestamps
        if df_processed.index.duplicated().any():
            duplicate_count = df_processed.index.duplicated().sum()
            issues.append(f"Found {duplicate_count} duplicate timestamps")
        
    except Exception as e:
        issues.append(f"Error during validation: {str(e)}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("Data quality validation: PASSED")
    else:
        print("Data quality validation: FAILED")
        for issue in issues:
            print(f"   - {issue}")
    
    return is_valid, issues

def print_system_info():
    """Print system information for debugging."""
    import platform
    import tensorflow as tf
    
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # GPU information
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
    else:
        print("No GPU devices found. Using CPU.")
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("psutil not available - cannot display memory information")
    
    print("=" * 50) 