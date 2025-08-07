"""
Model architectures and custom loss functions for Stock Price Predictor
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

from config import MODEL_CONFIG, RANDOM_SEED

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)

# === Custom Loss Functions ===
def direction_magnitude_loss(y_true, y_pred, alpha=10.0, beta=4.0):
    """
    Custom loss function that penalizes wrong direction (with alpha) and large return misses (with beta)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        alpha: strength of penalty for predicting wrong sign (direction)
        beta: strengthens loss on big true returns (percentage-like)
        
    Returns:
        Combined loss (fully differentiable)
    """
    mse = K.square(y_true - y_pred)
    # Directional penalty: relu activates only if signs are opposite
    dir_penalty = K.relu(-y_true * y_pred)
    # Magnitude penalty: weight errors by the size of true return
    mag_weight = 1.0 + beta * K.abs(y_true)
    # Final loss: weighted mse plus alpha*directional penalty
    loss = mag_weight * mse + alpha * dir_penalty
    return K.mean(loss)

# === Model Architecture ===
def build_simple_lstm_model(input_shape, config=None):
    """
    Build a super simple LSTM model for price prediction.
    
    Architecture:
    - Single LSTM layer 
    - One Dense output
    - No dropout, no BatchNorm, minimal complexity
    
    Args:
        input_shape: Shape of input sequences (time_steps, features)
        config: Model configuration dictionary (optional)
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = MODEL_CONFIG
    
    model = Sequential([
        LSTM(config['lstm_units'], input_shape=input_shape),
        Dense(1)  # Predict scaled price directly
    ])
    
    # Choose loss function
    if config['loss_function'] == 'direction_magnitude_loss':
        loss_fn = direction_magnitude_loss
    else:
        loss_fn = config['loss_function']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss_fn,
        metrics=config['metrics']
    )
    
    return model

def build_advanced_lstm_model(input_shape, config=None):
    """
    Build a more complex LSTM model with regularization.
    
    Architecture:
    - Stacked LSTM layers
    - Batch normalization
    - Dropout for regularization
    - Dense layers
    
    Args:
        input_shape: Shape of input sequences (time_steps, features)
        config: Model configuration dictionary (optional)
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = MODEL_CONFIG
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Choose loss function
    if config['loss_function'] == 'direction_magnitude_loss':
        loss_fn = direction_magnitude_loss
    else:
        loss_fn = config['loss_function']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss_fn,
        metrics=config['metrics']
    )
    
    return model

# === GPU Configuration ===
def setup_gpu():
    """Configure GPU settings for memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU setup complete. Found {len(gpus)} GPU(s)")
    else:
        print("No GPU found. Using CPU.")

# === Model Factory ===
def create_model(model_type='simple', input_shape=None, config=None):
    """
    Factory function to create different types of models.
    
    Args:
        model_type: 'simple' or 'advanced'
        input_shape: Shape of input sequences
        config: Model configuration
    
    Returns:
        Compiled model
    """
    if model_type == 'simple':
        return build_simple_lstm_model(input_shape, config)
    elif model_type == 'advanced':
        return build_advanced_lstm_model(input_shape, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 