"""
Training logic and callbacks for Stock Price Predictor
"""

import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import TRAINING_CONFIG, DIRECTORIES

class ImprovementCallback(tf.keras.callbacks.Callback):
    """Custom callback to track and display training improvements."""
    
    def on_train_begin(self, logs=None):
        """Initialize tracking variables at the start of training."""
        self.best_val_loss = float('inf')
        self.epoch_count = 0
        self.improvement_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """Track improvements at the end of each epoch."""
        self.epoch_count += 1
        current_val_loss = logs.get('val_loss', float('inf'))
        
        if current_val_loss < self.best_val_loss:
            improvement = self.best_val_loss - current_val_loss
            self.best_val_loss = current_val_loss
            self.improvement_count += 1
            print(f"EPOCH {self.epoch_count}: Model IMPROVED by {improvement:.6f} | Best Val Loss: {self.best_val_loss:.6f}")
        else:
            if self.best_val_loss != float('inf'):
                deterioration = current_val_loss - self.best_val_loss
                print(f"EPOCH {self.epoch_count}: No improvement | Current: {current_val_loss:.6f} (+{deterioration:.6f} from best)")
            else:
                print(f"EPOCH {self.epoch_count}: No improvement | Current: {current_val_loss:.6f}")
    
    def on_train_end(self, logs=None):
        """Display training summary at the end."""
        improvement_rate = (self.improvement_count / self.epoch_count) * 100
        print(f"\nTRAINING SUMMARY:")
        print(f"   Total epochs: {self.epoch_count}")
        print(f"   Epochs with improvement: {self.improvement_count}")
        print(f"   Improvement rate: {improvement_rate:.1f}%")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        
        if improvement_rate > 30:
            print("   Good learning progress!")
        elif improvement_rate > 15:
            print("   Moderate learning progress")
        else:
            print("   Poor learning progress - consider adjusting parameters")

def create_callbacks(config=None):
    """
    Create training callbacks.
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        List of callback objects
    """
    if config is None:
        config = TRAINING_CONFIG
    
    callbacks = [
        ImprovementCallback(),
        EarlyStopping(
            patience=config['early_stopping_patience'], 
            restore_best_weights=True, 
            monitor="val_loss",
            verbose=1
        ),
        ReduceLROnPlateau(
            patience=config['lr_reduction_patience'], 
            factor=config['lr_reduction_factor'], 
            min_lr=config['min_learning_rate'], 
            monitor="val_loss", 
            verbose=1
        )
    ]
    
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, config=None, verbose=1):
    """
    Train the model with the given data.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Training configuration
        verbose: Training verbosity level
    
    Returns:
        Training history
    """
    if config is None:
        config = TRAINING_CONFIG
    
    print(f"Starting training with configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Early stopping patience: {config['early_stopping_patience']}")
    print(f"  Learning rate reduction patience: {config['lr_reduction_patience']}")
    print()
    
    callbacks = create_callbacks(config)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history

def save_model(model, model_name=None, prediction_steps=1):
    """
    Save the trained model with timestamp.
    
    Args:
        model: Trained Keras model
        model_name: Custom model name (optional)
        prediction_steps: Number of prediction steps for filename
    
    Returns:
        Path where model was saved
    """
    # Create model_checkpoints directory if it doesn't exist
    checkpoint_dir = DIRECTORIES['model_checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is None:
        model_filename = f'price_predictor_{prediction_steps}steps_{timestamp}.h5'
    else:
        model_filename = f'{model_name}_{timestamp}.h5'
    
    model_path = os.path.join(checkpoint_dir, model_filename)
    
    # Save model
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
    
    return model_path

def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from '{model_path}'")
    return model

def get_training_summary(history):
    """
    Get a summary of training results.
    
    Args:
        history: Training history object
    
    Returns:
        Dictionary with training summary
    """
    if history is None:
        return {}
    
    final_epoch = len(history.history['loss'])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    # Check if learning rate was reduced
    lr_reductions = 0
    if 'lr' in history.history:
        initial_lr = history.history['lr'][0]
        final_lr = history.history['lr'][-1]
        lr_reductions = sum(1 for i in range(1, len(history.history['lr'])) 
                           if history.history['lr'][i] < history.history['lr'][i-1])
    else:
        initial_lr = final_lr = None
    
    summary = {
        'total_epochs': final_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'initial_lr': initial_lr,
        'final_lr': final_lr,
        'lr_reductions': lr_reductions
    }
    
    return summary

def print_training_summary(history):
    """
    Print a formatted training summary.
    
    Args:
        history: Training history object
    """
    summary = get_training_summary(history)
    
    if not summary:
        print("No training history available.")
        return
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Total epochs trained: {summary['total_epochs']}")
    print(f"Best validation loss: {summary['best_val_loss']:.6f} (epoch {summary['best_epoch']})")
    print(f"Final train loss: {summary['final_train_loss']:.6f}")
    print(f"Final validation loss: {summary['final_val_loss']:.6f}")
    
    if summary['initial_lr'] is not None:
        print(f"Learning rate: {summary['initial_lr']:.2e} → {summary['final_lr']:.2e}")
        if summary['lr_reductions'] > 0:
            print(f"Learning rate reductions: {summary['lr_reductions']}")
    
    # Performance assessment
    improvement = (summary['final_val_loss'] - summary['best_val_loss']) / summary['best_val_loss']
    if improvement < 0.01:  # Less than 1% worse than best
        print("Status: Converged well")
    elif improvement < 0.05:  # Less than 5% worse than best
        print("Status: Good convergence")
    else:
        print("Status: May need more training or hyperparameter tuning")
    
    print("="*50) 