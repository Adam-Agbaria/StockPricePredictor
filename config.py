"""
Configuration settings for the Stock Price Predictor
"""

import os

# === GPU Configuration ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# === Data Configuration ===
DATA_CONFIG = {
    'csv_path': 'data/nq-1m.csv',
    'sequence_length': 20,
    'test_ratio': 0.2,
    'prediction_steps': 5,  # Predict N steps ahead
    'features': ['close', 'volume', 'hl_range', 'close_ma', 'vol_ma'],
    'target_column': 'close'
}

# === Model Configuration ===
MODEL_CONFIG = {
    'lstm_units': 64,
    'learning_rate': 0.00005,
    'loss_function': 'mse',  
    'metrics': ['mae']
}

# === Training Configuration ===
TRAINING_CONFIG = {
    'epochs': 60,
    'batch_size': 64,
    'early_stopping_patience': 10,
    'lr_reduction_patience': 30,
    'lr_reduction_factor': 0.5,
    'min_learning_rate': 1e-7,
    'validation_split': 0.2
}

# === Evaluation Configuration ===
EVALUATION_CONFIG = {
    'plot_samples': 500,  # Number of samples to show in plot
    'save_plots': True,
    'save_reports': True
}

# === Directory Configuration ===
DIRECTORIES = {
    'model_checkpoints': 'model_checkpoints',
    'visualizations': 'visualizations_1m',
    'evaluations': 'evaluation_1m'
}

# === Random Seeds ===
RANDOM_SEED = 42

# === Prediction Horizon Analysis ===
HORIZON_ANALYSIS = {
    'horizons': [10, 30, 60, 120],
    'min_avg_delta': 0.05,
    'max_zero_percentage': 50
} 