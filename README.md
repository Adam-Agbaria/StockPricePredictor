# Stock Price Predictor - Modular Architecture

A clean, modular stock price prediction system using LSTM neural networks, organized into focused modules for better maintainability and extensibility.

##  Project Structure

```
├── config.py              # Configuration settings and parameters
├── models.py               # Model architectures and custom loss functions
├── data_processing.py      # Data loading, preprocessing, and feature engineering
├── training.py             # Training logic, callbacks, and model management
├── evaluation.py           # Evaluation metrics, plotting, and reporting
├── utils.py                # Utility functions and data analysis tools
├── main.py                 # Main execution script
├── README.md               # This file
└── data/                   # Data directory
    └── nq-1m.csv           # Your stock data file
```

##  Quick Start

### Basic Usage
```bash
# Run the complete training pipeline
python main.py

# Or explicitly specify full mode
python main.py full
```

### Test Mode (Quick Testing)
```bash
# Run with minimal epochs for quick testing
python main.py test
```

### Analysis Mode (Data Analysis Only)
```bash
# Run data analysis without training
python main.py analyze
```

## Module Details

### `config.py`
**Central configuration management**
- Data configuration (file paths, features, prediction settings)
- Model configuration (architecture, learning rate, loss function)
- Training configuration (epochs, batch size, callbacks)
- Directory paths and evaluation settings
- Random seeds for reproducibility

### `models.py`
**Model architectures and loss functions**
- `build_simple_lstm_model()` - Simple LSTM architecture
- `build_advanced_lstm_model()` - Complex LSTM with regularization
- `direction_magnitude_loss()` - Custom loss function
- `create_model()` - Model factory function
- GPU setup and configuration

### `data_processing.py`
**Complete data pipeline**
- `load_raw_data()` - CSV loading and parsing
- `add_technical_features()` - Technical indicator creation
- `create_target_variable()` - Target variable generation
- `build_sequences()` - LSTM sequence creation
- `scale_data()` - Feature and target scaling
- `load_and_preprocess()` - Complete preprocessing pipeline

### `training.py`
**Training management**
- `ImprovementCallback` - Custom training progress tracking
- `create_callbacks()` - Callback setup
- `train_model()` - Model training with monitoring
- `save_model()` / `load_model()` - Model persistence
- Training summary and analysis

### `evaluation.py`
**Comprehensive evaluation**
- `calculate_basic_metrics()` - MSE, RMSE, MAE, MAPE, correlation
- `calculate_hit_rates()` - Prediction accuracy within thresholds
- `calculate_directional_accuracy()` - Price movement direction accuracy
- `evaluate_model()` - Complete model evaluation
- `plot_predictions()` - Visualization generation
- `save_evaluation_report()` - Detailed report creation

### `utils.py`
**Analysis and utility functions**
- `analyze_prediction_horizons()` - Optimal time horizon analysis
- `calculate_data_statistics()` - Dataset statistics
- `validate_data_quality()` - Data quality checks
- `print_system_info()` - System information display

### `main.py`
**Orchestration and execution**
- Complete training pipeline orchestration
- Error handling and recovery
- Multiple execution modes (full/test/analyze)
- Command-line interface

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Change data source
DATA_CONFIG['csv_path'] = 'your_data.csv'

# Adjust prediction horizon
DATA_CONFIG['prediction_steps'] = 30  # Predict 30 minutes ahead

# Modify model architecture
MODEL_CONFIG['lstm_units'] = 128
MODEL_CONFIG['learning_rate'] = 0.001

# Training parameters
TRAINING_CONFIG['epochs'] = 100
TRAINING_CONFIG['batch_size'] = 32
```

## Features

### Data Processing
- Automatic CSV parsing with datetime indexing
- Technical indicator generation (moving averages, ranges)
- Robust data cleaning and validation
- Configurable sequence generation for LSTM input
- MinMaxScaler for features and targets

### Model Architecture
- Simple LSTM architecture (default)
- Advanced LSTM with regularization (optional)
- Custom directional loss functions
- GPU acceleration support
- Model factory pattern for flexibility

### Training
- Progress tracking with custom callbacks
- Early stopping and learning rate reduction
- Comprehensive training summaries
- Model checkpointing with timestamps
- Configurable training parameters

### Evaluation
- Comprehensive metrics (MAPE, correlation, hit rates)
- Directional accuracy analysis
- Prediction bias assessment
- Automated visualization generation
- Detailed evaluation reports
- Performance categorization (Exceptional/Good/Moderate/Poor)

### Utilities
- Prediction horizon optimization
- Data quality validation
- System information display
- Statistical analysis tools

## Output Files

The system generates timestamped outputs:

```
model_checkpoints/
├── simple_price_predictor_1steps_20231201_143022.h5

evaluation_1m/
├── evaluation_report_20231201_143022.txt
```

## Advanced Usage

### Custom Model Architecture
```python
from models import create_model, MODEL_CONFIG

# Use advanced model
model = create_model(
    model_type='advanced',
    input_shape=(20, 5),
    config=MODEL_CONFIG
)
```

### Custom Training Configuration
```python
from training import train_model

custom_config = {
    'epochs': 200,
    'batch_size': 16,
    'early_stopping_patience': 20
}

history = train_model(model, X_train, y_train, X_test, y_test, custom_config)
```

### Data Analysis
```python
from utils import analyze_prediction_horizons, calculate_data_statistics

# Analyze optimal prediction horizons
results = analyze_prediction_horizons('your_data.csv')

# Get dataset statistics
stats = calculate_data_statistics('your_data.csv')
```

##  Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

##  Benefits of Modular Architecture

1. **Maintainability** - Each module has a single responsibility
2. **Reusability** - Functions can be imported and used independently
3. **Testability** - Individual modules can be tested in isolation
4. **Extensibility** - Easy to add new features without affecting other parts
5. **Configuration** - Centralized configuration management
6. **Error Handling** - Better error isolation and recovery
7. **Documentation** - Clear separation of concerns and functionality

##  Next Steps

- Add more technical indicators in `data_processing.py`
- Implement additional model architectures in `models.py`
- Add cross-validation in `training.py`
- Create more sophisticated evaluation metrics in `evaluation.py`
- Add hyperparameter optimization utilities in `utils.py` 