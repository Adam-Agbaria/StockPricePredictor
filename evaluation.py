"""
Evaluation metrics, plotting, and reporting for Stock Price Predictor
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import DIRECTORIES, EVALUATION_CONFIG

def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate basic prediction metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with basic metrics
    """
    # Ensure arrays are flattened
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic error metrics
    errors = y_pred_flat - y_true_flat
    abs_errors = np.abs(errors)
    percentage_errors = (errors / y_true_flat) * 100
    abs_percentage_errors = np.abs(percentage_errors)
    
    # Main metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(abs_percentage_errors)
    
    # Additional error statistics
    max_error = np.max(abs_errors)
    min_error = np.min(abs_errors)
    std_error = np.std(errors)
    median_ae = np.median(abs_errors)
    percentile_95_error = np.percentile(abs_errors, 95)
    mean_pe = np.mean(percentage_errors)  # Signed percentage error (bias)
    
    # Correlation and R-squared
    corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    r_squared = corr ** 2
    
    # Prediction bias
    prediction_bias = np.mean(y_pred_flat) - np.mean(y_true_flat)
    bias_percentage = (prediction_bias / np.mean(y_true_flat)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'correlation': corr,
        'r_squared': r_squared,
        'max_error': max_error,
        'min_error': min_error,
        'std_error': std_error,
        'median_ae': median_ae,
        'percentile_95_error': percentile_95_error,
        'mean_pe': mean_pe,
        'prediction_bias': prediction_bias,
        'bias_percentage': bias_percentage
    }

def calculate_hit_rates(y_true, y_pred):
    """
    Calculate hit rates for different error thresholds.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with hit rates
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    percentage_errors = np.abs((y_pred_flat - y_true_flat) / y_true_flat) * 100
    
    hit_rates = {
        'hit_rate_1pct': np.mean(percentage_errors <= 1.0) * 100,
        'hit_rate_2pct': np.mean(percentage_errors <= 2.0) * 100,
        'hit_rate_5pct': np.mean(percentage_errors <= 5.0) * 100
    }
    
    return hit_rates

def calculate_directional_accuracy(y_true, y_pred, current_prices):
    """
    Calculate directional accuracy metrics.
    
    Args:
        y_true: True future prices
        y_pred: Predicted future prices
        current_prices: Current prices for movement calculation
    
    Returns:
        Dictionary with directional accuracy metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate actual and predicted price movements
    actual_movement = y_true_flat - current_prices
    predicted_movement = y_pred_flat - current_prices
    
    # Calculate directional accuracy (same direction of movement)
    actual_direction = np.sign(actual_movement)
    predicted_direction = np.sign(predicted_movement)
    
    true_directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Break down by movement type
    up_movements = actual_direction > 0
    down_movements = actual_direction < 0
    
    # True directional accuracy breakdown
    up_correct = np.sum((actual_direction == predicted_direction) & up_movements)
    up_total = np.sum(up_movements)
    down_correct = np.sum((actual_direction == predicted_direction) & down_movements)
    down_total = np.sum(down_movements)
    
    true_up_accuracy = (up_correct / up_total * 100) if up_total > 0 else 0
    true_down_accuracy = (down_correct / down_total * 100) if down_total > 0 else 0
    
    # Statistical directional accuracy (vs mean)
    true_mean = np.mean(y_true_flat)
    pred_mean = np.mean(y_pred_flat)
    
    true_above_mean = y_true_flat > true_mean
    pred_above_mean = y_pred_flat > pred_mean
    
    stat_directional_accuracy = np.mean(true_above_mean == pred_above_mean) * 100
    
    stat_up_total = np.sum(true_above_mean)
    stat_down_total = np.sum(~true_above_mean)
    stat_up_correct = np.sum((true_above_mean == pred_above_mean) & true_above_mean)
    stat_down_correct = np.sum((true_above_mean == pred_above_mean) & ~true_above_mean)
    
    stat_up_accuracy = (stat_up_correct / stat_up_total * 100) if stat_up_total > 0 else 0
    stat_down_accuracy = (stat_down_correct / stat_down_total * 100) if stat_down_total > 0 else 0
    
    return {
        'true_directional_accuracy': true_directional_accuracy,
        'true_up_accuracy': true_up_accuracy,
        'true_down_accuracy': true_down_accuracy,
        'true_up_correct': up_correct,
        'true_up_total': up_total,
        'true_down_correct': down_correct,
        'true_down_total': down_total,
        'stat_directional_accuracy': stat_directional_accuracy,
        'stat_up_accuracy': stat_up_accuracy,
        'stat_down_accuracy': stat_down_accuracy,
        'stat_up_correct': stat_up_correct,
        'stat_up_total': stat_up_total,
        'stat_down_correct': stat_down_correct,
        'stat_down_total': stat_down_total
    }

def evaluate_model(model, X_test, y_test, price_scaler, feature_scaler):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        price_scaler: Price scaler for inverse transform
        feature_scaler: Feature scaler for inverse transform
    
    Returns:
        Tuple of (y_true_price, y_pred_price, metrics)
    """
    # Predict scaled prices
    y_pred_scaled = model.predict(X_test)
    
    # Convert back to actual prices
    y_pred_price = price_scaler.inverse_transform(y_pred_scaled)
    y_true_price = price_scaler.inverse_transform(y_test)
    
    # Extract current prices from the last time step of each sequence
    current_prices_scaled = X_test[:, -1, 0].reshape(-1, 1)
    last_features_scaled = X_test[:, -1, :]
    last_features_unscaled = feature_scaler.inverse_transform(last_features_scaled)
    current_prices = last_features_unscaled[:, 0]
    
    # Calculate all metrics
    basic_metrics = calculate_basic_metrics(y_true_price, y_pred_price)
    hit_rates = calculate_hit_rates(y_true_price, y_pred_price)
    directional_metrics = calculate_directional_accuracy(
        y_true_price, y_pred_price, current_prices
    )
    
    # Combine all metrics
    metrics = {**basic_metrics, **hit_rates, **directional_metrics}
    
    # Display results
    print_evaluation_results(metrics)
    
    return y_true_price, y_pred_price, metrics

def print_evaluation_results(metrics):
    """
    Print comprehensive evaluation results to console.
    
    Args:
        metrics: Dictionary with all evaluation metrics
    """
    print(f"\nComprehensive Price Prediction Evaluation:")
    print(f"\nBasic Metrics:")
    print(f"   MSE:         {metrics['mse']:.4f}")
    print(f"   RMSE:        {metrics['rmse']:.4f}")
    print(f"   MAE:         {metrics['mae']:.4f}")
    print(f"   MAPE:        {metrics['mape']:.2f}%")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    print(f"   R-squared:   {metrics['r_squared']:.4f}")
    
    print(f"\nHit Rate Analysis:")
    print(f"   Within 1%:   {metrics['hit_rate_1pct']:.2f}%")
    print(f"   Within 2%:   {metrics['hit_rate_2pct']:.2f}%")
    print(f"   Within 5%:   {metrics['hit_rate_5pct']:.2f}%")
    
    print(f"\nError Statistics:")
    print(f"   Max Error:   {metrics['max_error']:.4f}")
    print(f"   Min Error:   {metrics['min_error']:.4f}")
    print(f"   Std Error:   {metrics['std_error']:.4f}")
    print(f"   Median AE:   {metrics['median_ae']:.4f}")
    print(f"   95th Pct:    {metrics['percentile_95_error']:.4f}")
    
    print(f"\nPrediction Bias:")
    print(f"   Bias:        {metrics['prediction_bias']:.4f}")
    print(f"   Bias %:      {metrics['bias_percentage']:.2f}%")
    if abs(metrics['bias_percentage']) < 1:
        print(f"   Assessment: Low bias")
    elif abs(metrics['bias_percentage']) < 3:
        print(f"   Assessment: Moderate bias")
    else:
        print(f"   Assessment: High bias")

def plot_predictions(y_true, y_pred, title="Price Prediction Results"):
    """
    Create and save prediction plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    
    Returns:
        Path where plot was saved
    """
    if not EVALUATION_CONFIG['save_plots']:
        return None
    
    # Create visualizations directory if it doesn't exist
    viz_dir = DIRECTORIES['visualizations']
    os.makedirs(viz_dir, exist_ok=True)
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Limit samples for plotting if specified
    plot_samples = min(EVALUATION_CONFIG['plot_samples'], len(y_true_flat))
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_true_flat[:plot_samples], label='Actual Price', alpha=0.7)
    plt.plot(y_pred_flat[:plot_samples], label='Predicted Price', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Add timestamp to plot name for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'price_prediction_{timestamp}.png'
    plot_path = os.path.join(viz_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved plot: {plot_path}")
    
    return plot_path

def save_evaluation_report(metrics, model_config, data_config):
    """
    Save comprehensive evaluation report to file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary
    
    Returns:
        Path where report was saved
    """
    if not EVALUATION_CONFIG['save_reports']:
        return None
    
    # Create evaluation directory if it doesn't exist
    eval_dir = DIRECTORIES['evaluations']
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f'evaluation_report_{timestamp}.txt'
    eval_path = os.path.join(eval_dir, eval_filename)
    
    # Write evaluation report
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("STOCK PRICE PREDICTION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Timestamp and general info
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: Simple LSTM Price Predictor\n\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"LSTM Units: {model_config['lstm_units']}\n")
        f.write(f"Learning Rate: {model_config['learning_rate']}\n")
        f.write(f"Epochs Trained: {model_config['epochs']}\n")
        f.write(f"Batch Size: {model_config['batch_size']}\n\n")
        
        # Data Configuration
        f.write("DATA CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Prediction Steps: {data_config['prediction_steps']} steps ahead\n")
        f.write(f"Sequence Length: {data_config['sequence_length']} time steps\n")
        f.write(f"Test Ratio: {data_config['test_ratio']:.1%}\n")
        f.write(f"Features: {', '.join(data_config['features'])}\n\n")
        
        # Basic Performance Metrics
        f.write("BASIC PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Squared Error (MSE):      {metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")
        f.write(f"Correlation Coefficient:       {metrics['correlation']:.4f}\n")
        f.write(f"R-squared:                     {metrics['r_squared']:.4f}\n\n")
        
        # Hit Rate Analysis
        f.write("HIT RATE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Predictions within 1%:         {metrics['hit_rate_1pct']:.2f}%\n")
        f.write(f"Predictions within 2%:         {metrics['hit_rate_2pct']:.2f}%\n")
        f.write(f"Predictions within 5%:         {metrics['hit_rate_5pct']:.2f}%\n\n")
        
        # Error Distribution Statistics
        f.write("ERROR DISTRIBUTION STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Maximum Error:                 {metrics['max_error']:.4f}\n")
        f.write(f"Minimum Error:                 {metrics['min_error']:.4f}\n")
        f.write(f"Standard Deviation of Errors:  {metrics['std_error']:.4f}\n")
        f.write(f"Median Absolute Error:         {metrics['median_ae']:.4f}\n")
        f.write(f"95th Percentile Error:         {metrics['percentile_95_error']:.4f}\n")
        f.write(f"Mean Percentage Error (bias):  {metrics['mean_pe']:.2f}%\n\n")
        
        # Prediction Bias Analysis
        f.write("PREDICTION BIAS ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Prediction Bias (absolute):    {metrics['prediction_bias']:.4f}\n")
        f.write(f"Prediction Bias (percentage):  {metrics['bias_percentage']:.2f}%\n")
        if abs(metrics['bias_percentage']) < 1:
            f.write("Bias Assessment:               Low bias - predictions well centered\n")
        elif abs(metrics['bias_percentage']) < 3:
            f.write("Bias Assessment:               Moderate bias - slight systematic error\n")
        else:
            f.write("Bias Assessment:               High bias - systematic over/under prediction\n")
        f.write("\n")
        
        # Comprehensive Performance Assessment
        f.write("COMPREHENSIVE PERFORMANCE ASSESSMENT:\n")
        f.write("-" * 50 + "\n")
        
        # MAPE Assessment
        if metrics['mape'] < 1.0:
            f.write("MAPE Assessment:               EXCEPTIONAL (< 1%) - Extremely accurate\n")
        elif metrics['mape'] < 2.0:
            f.write("MAPE Assessment:               EXCELLENT (< 2%) - Very accurate predictions\n")
        elif metrics['mape'] < 5.0:
            f.write("MAPE Assessment:               GOOD (< 5%) - Good prediction accuracy\n")
        elif metrics['mape'] < 10.0:
            f.write("MAPE Assessment:               MODERATE (< 10%) - Acceptable accuracy\n")
        else:
            f.write("MAPE Assessment:               POOR (> 10%) - Low prediction accuracy\n")
            
        # Correlation Assessment
        if metrics['correlation'] > 0.9:
            f.write("Correlation Assessment:        EXCEPTIONAL (> 0.9) - Very strong relationship\n")
        elif metrics['correlation'] > 0.8:
            f.write("Correlation Assessment:        STRONG (> 0.8) - Strong relationship captured\n")
        elif metrics['correlation'] > 0.6:
            f.write("Correlation Assessment:        MODERATE (> 0.6) - Moderate relationship\n")
        elif metrics['correlation'] > 0.3:
            f.write("Correlation Assessment:        WEAK (> 0.3) - Weak relationship\n")
        else:
            f.write("Correlation Assessment:        VERY WEAK (< 0.3) - Poor pattern learning\n")
            
        # Hit Rate Assessment
        if metrics['hit_rate_5pct'] > 80:
            f.write("Hit Rate Assessment:           EXCELLENT - High precision predictions\n")
        elif metrics['hit_rate_5pct'] > 60:
            f.write("Hit Rate Assessment:           GOOD - Reasonable precision\n")
        elif metrics['hit_rate_5pct'] > 40:
            f.write("Hit Rate Assessment:           MODERATE - Acceptable precision\n")
        else:
            f.write("Hit Rate Assessment:           POOR - Low precision predictions\n")
            
        # Overall Model Assessment
        f.write("\n" + "=" * 50 + "\n")
        f.write("OVERALL MODEL ASSESSMENT:\n")
        f.write("=" * 50 + "\n")
        
        # Calculate overall score
        mape_score = max(0, min(100, (10 - metrics['mape']) * 10))
        corr_score = max(0, metrics['correlation'] * 100)
        hit_score = max(0, metrics['hit_rate_5pct'])
        
        overall_score = (mape_score + corr_score + hit_score) / 3
        
        f.write(f"Overall Score: {overall_score:.1f}/100\n\n")
        
        if overall_score >= 80:
            f.write("EXCEPTIONAL MODEL - Ready for production use\n")
        elif overall_score >= 65:
            f.write("GOOD MODEL - Suitable for trading with proper risk management\n")
        elif overall_score >= 50:
            f.write("MODERATE MODEL - Requires improvement or additional validation\n")
        else:
            f.write("POOR MODEL - Needs significant improvement before use\n")
            
        f.write("\nRecommendations:\n")
        if metrics['mape'] > 5:
            f.write("• Improve prediction accuracy (high MAPE)\n")
        if metrics['correlation'] < 0.6:
            f.write("• Enhance feature engineering or model complexity\n")
        if abs(metrics['bias_percentage']) > 2:
            f.write("• Address prediction bias issues\n")
        if metrics['hit_rate_5pct'] < 50:
            f.write("• Improve prediction precision\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Evaluation report saved: {eval_path}")
    return eval_path 