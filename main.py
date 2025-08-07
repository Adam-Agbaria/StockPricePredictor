"""
Main execution script for Stock Price Predictor
"""

import numpy as np
from datetime import datetime

# Import all our modules
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, RANDOM_SEED
from models import create_model, setup_gpu
from data_processing import load_and_preprocess
from training import train_model, save_model, print_training_summary
from evaluation import evaluate_model, plot_predictions, save_evaluation_report
from utils import analyze_prediction_horizons, print_system_info, validate_data_quality

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

def main():
    """Main training and evaluation pipeline."""
    
    print("STOCK PRICE PREDICTOR")
    print("=" * 60)
    
    # Print system information
    print_system_info()
    print()
    
    # Setup GPU
    setup_gpu()
    print()
    
    # Configuration
    csv_path = DATA_CONFIG['csv_path']
    prediction_steps = DATA_CONFIG['prediction_steps']
    
    print(f"SIMPLE PRICE PREDICTION")
    print(f"   Predicting price {prediction_steps} steps ahead ({prediction_steps} minutes)")
    print(f"   Super simple architecture: LSTM({MODEL_CONFIG['lstm_units']}) → Dense(1)")
    print(f"   Using MinMaxScaler for both features and targets")
    print(f"   Data source: {csv_path}\n")
    
    # Validate data quality
    print("VALIDATING DATA QUALITY...")
    is_valid, issues = validate_data_quality(csv_path)
    if not is_valid:
        print("WARNING: Data quality issues detected. Proceeding anyway...")
        for issue in issues:
            print(f"   - {issue}")
    print()
    
    # Optional: Analyze prediction horizons
    print("ANALYZING PREDICTION HORIZONS...")
    horizon_results = analyze_prediction_horizons(csv_path)
    print()
    
    # Load and preprocess data
    print("LOADING AND PREPROCESSING DATA...")
    try:
        (X_train, X_test, y_train, y_test, 
         price_scaler, feature_scaler) = load_and_preprocess(
            csv_path=csv_path,
            sequence_length=DATA_CONFIG['sequence_length'],
            test_ratio=DATA_CONFIG['test_ratio'],
            prediction_steps=prediction_steps
        )
    except Exception as e:
        print(f"ERROR: Failed to load and preprocess data: {e}")
        return
    
    # Build model
    print("BUILDING MODEL...")
    try:
        model = create_model(
            model_type='simple',  # Use 'advanced' for more complex model
            input_shape=X_train.shape[1:],
            config=MODEL_CONFIG
        )
        print(f"Model created successfully")
        print(f"   Architecture: {MODEL_CONFIG['lstm_units']} LSTM units → Dense(1)")
        print(f"   Loss function: {MODEL_CONFIG['loss_function']}")
        print(f"   Learning rate: {MODEL_CONFIG['learning_rate']}")
        print()
        
        # Display model summary
        model.summary()
        print()
    except Exception as e:
        print(f"ERROR: Failed to build model: {e}")
        return
    
    # Train model
    print("STARTING TRAINING...")
    try:
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            config=TRAINING_CONFIG
        )
        print_training_summary(history)
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return
    
    # Evaluate model
    print("EVALUATING MODEL...")
    try:
        y_true, y_pred, metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            price_scaler=price_scaler,
            feature_scaler=feature_scaler
        )
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return
    
    # Generate plots
    print("\nGENERATING VISUALIZATIONS...")
    try:
        plot_path = plot_predictions(
            y_true=y_true, 
            y_pred=y_pred,
            title="Simple Price Prediction: Predicted vs Actual Prices"
        )
    except Exception as e:
        print(f"WARNING: Failed to generate plots: {e}")
        plot_path = None
    
    # Save model
    print("SAVING MODEL...")
    try:
        model_path = save_model(
            model=model,
            model_name='simple_price_predictor',
            prediction_steps=prediction_steps
        )
    except Exception as e:
        print(f"WARNING: Failed to save model: {e}")
        model_path = None
    
    # Save evaluation report
    print("GENERATING EVALUATION REPORT...")
    try:
        # Prepare configuration for report
        model_config_for_report = {
            'lstm_units': MODEL_CONFIG['lstm_units'],
            'learning_rate': MODEL_CONFIG['learning_rate'],
            'epochs': TRAINING_CONFIG['epochs'],
            'batch_size': TRAINING_CONFIG['batch_size']
        }
        
        data_config_for_report = {
            'prediction_steps': prediction_steps,
            'sequence_length': DATA_CONFIG['sequence_length'],
            'test_ratio': DATA_CONFIG['test_ratio'],
            'csv_path': csv_path,
            'features': DATA_CONFIG['features']
        }
        
        report_path = save_evaluation_report(
            metrics=metrics,
            model_config=model_config_for_report,
            data_config=data_config_for_report
        )
    except Exception as e:
        print(f"WARNING: Failed to save evaluation report: {e}")
        report_path = None
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model performance (MAPE): {metrics['mape']:.2f}%")
    print(f"Model performance (Correlation): {metrics['correlation']:.4f}")
    print(f"Model performance (Hit Rate 5%): {metrics['hit_rate_5pct']:.1f}%")
    
    if model_path:
        print(f"Model saved: {model_path}")
    if plot_path:
        print(f"Visualization saved: {plot_path}")
    if report_path:
        print(f"Evaluation report saved: {report_path}")
    
    print("=" * 60)

def run_quick_test():
    """Run a quick test with minimal epochs for debugging."""
    print("RUNNING QUICK TEST MODE")
    print("=" * 40)
    
    # Override configuration for quick test
    test_config = TRAINING_CONFIG.copy()
    test_config['epochs'] = 2
    test_config['early_stopping_patience'] = 1
    
    model_config = MODEL_CONFIG.copy()
    model_config['lstm_units'] = 32  # Smaller model for quick test
    
    # Setup GPU
    setup_gpu()
    
    # Load minimal data
    (X_train, X_test, y_train, y_test, 
     price_scaler, feature_scaler) = load_and_preprocess()
    
    # Use only a subset of data for quick test
    subset_size = min(1000, len(X_train))
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    X_test = X_test[:100]
    y_test = y_test[:100]
    
    print(f"Using subset: Train={len(X_train)}, Test={len(X_test)}")
    
    # Build and train model
    model = create_model('simple', X_train.shape[1:], model_config)
    history = train_model(model, X_train, y_train, X_test, y_test, test_config)
    
    # Quick evaluation
    y_true, y_pred, metrics = evaluate_model(model, X_test, y_test, price_scaler, feature_scaler)
    
    print(f"\nQuick test results:")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    print("Quick test completed!")

def run_data_analysis_only():
    """Run only data analysis without training."""
    print("DATA ANALYSIS MODE")
    print("=" * 40)
    
    from utils import calculate_data_statistics
    
    # Analyze data
    stats = calculate_data_statistics()
    
    # Analyze horizons
    horizon_results = analyze_prediction_horizons()
    
    # Validate data quality
    is_valid, issues = validate_data_quality()
    
    print("\nData analysis completed!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for different modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'test':
            run_quick_test()
        elif mode == 'analyze':
            run_data_analysis_only()
        elif mode == 'full':
            main()
        else:
            print("Usage: python main.py [full|test|analyze]")
            print("  full:    Run complete training pipeline (default)")
            print("  test:    Run quick test with minimal epochs")
            print("  analyze: Run data analysis only")
    else:
        # Default: run full pipeline
        main() 