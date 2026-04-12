"""
ML Model Manager with caching and efficient prediction
Handles model training, predictions, and performance metrics.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Efficient ML model management with caching and versioning."""
    
    def __init__(self, cache_dir: str = './model_cache'):
        """Initialize model manager with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.prediction_cache = {}
        self.model_metadata = {}
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                               sequence_length: int = 10,
                               model_name: str = "lr_model") -> 'ModelManager':
        """
        Train Linear Regression model efficiently.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sequence_length: Length of input sequences
            model_name: Name to save model under
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Linear Regression model: {model_name}")
        
        # Flatten sequences if needed
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Train model
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train_flat, y_train)
        
        self.models[model_name] = model
        
        # Store metadata
        self.model_metadata[model_name] = {
            'type': 'LinearRegression',
            'trained_at': datetime.now().isoformat(),
            'sequence_length': sequence_length,
            'training_samples': len(X_train),
            'features': X_train_flat.shape[1]
        }
        
        logger.info(f"Model {model_name} trained successfully")
        return self
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    model_name: str = "lr_model",
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training.
        
        Args:
            X: Features
            y: Targets
            model_name: Model identifier
            test_size: Train/test split ratio
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Flatten if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X_flat)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Store scalers
        self.scalers[f'{model_name}_X'] = scaler_X
        self.scalers[f'{model_name}_y'] = scaler_y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
        
        logger.info(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def predict(self, X: np.ndarray, model_name: str = "lr_model",
               inverse_scale: bool = True) -> np.ndarray:
        """
        Make predictions with caching.
        
        Args:
            X: Input features
            model_name: Model to use
            inverse_scale: Whether to inverse scale predictions
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Flatten if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Make predictions
        predictions = self.models[model_name].predict(X_flat)
        
        # Inverse scale if needed
        if inverse_scale and f'{model_name}_y' in self.scalers:
            scaler_y = self.scalers[f'{model_name}_y']
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def predict_next_values(self, last_sequence: np.ndarray, num_steps: int = 10,
                           model_name: str = "lr_model") -> np.ndarray:
        """
        Predict next num_steps values using sliding window.
        
        Args:
            last_sequence: Last known sequence (1D array of features)
            num_steps: Number of steps to predict
            model_name: Model to use
            
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        predictions = []
        
        # Ensure last_sequence is 1D and flatten if needed
        if len(last_sequence.shape) > 1:
            current_sequence = last_sequence.flatten()
        else:
            current_sequence = last_sequence.copy()
        
        for _ in range(num_steps):
            # Reshape for model input: (1, num_features)
            input_data = current_sequence.reshape(1, -1)
            
            # Ensure input has correct shape for model
            if input_data.shape[1] == 0:
                logger.error(f"Invalid input shape: {input_data.shape}")
                break
            
            # Predict
            next_value = self.models[model_name].predict(input_data)[0]
            
            # Inverse scale if needed
            if f'{model_name}_y' in self.scalers:
                scaler_y = self.scalers[f'{model_name}_y']
                next_value = scaler_y.inverse_transform([[next_value]])[0, 0]
            
            predictions.append(next_value)
            
            # Update sequence (sliding window): remove first element, add prediction
            # Scale prediction back for next iteration
            scaled_pred = next_value / 100 if next_value > 1 else next_value
            current_sequence = np.append(current_sequence[1:], scaled_pred)
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = "lr_model") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Model to evaluate
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test, model_name, inverse_scale=True)
        
        # Inverse scale y_test if needed
        if f'{model_name}_y' in self.scalers:
            scaler_y = self.scalers[f'{model_name}_y']
            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_test_inv = y_test
        
        metrics = {
            'mse': mean_squared_error(y_test_inv, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test_inv, predictions)),
            'mae': mean_absolute_error(y_test_inv, predictions),
            'r2': r2_score(y_test_inv, predictions)
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"Model {model_name} metrics: MSE={metrics['mse']:.4f}, "
                   f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        return metrics
    
    def save_model(self, model_name: str = "lr_model"):
        """Save model and scalers to cache."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Save model
        model_path = self.cache_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        # Save scalers
        for scaler_name in [f'{model_name}_X', f'{model_name}_y']:
            if scaler_name in self.scalers:
                scaler_path = self.cache_dir / f"{scaler_name}.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[scaler_name], f)
        
        # Save metadata
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata[model_name], f, indent=2)
        
        logger.info(f"Model {model_name} saved to {self.cache_dir}")
    
    def load_model(self, model_name: str = "lr_model"):
        """Load model and scalers from cache."""
        # Load model
        model_path = self.cache_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise ValueError(f"Model file {model_path} not found")
        
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        
        # Load scalers
        for scaler_name in [f'{model_name}_X', f'{model_name}_y']:
            scaler_path = self.cache_dir / f"{scaler_name}.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers[scaler_name] = pickle.load(f)
        
        # Load metadata
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_name] = json.load(f)
        
        logger.info(f"Model {model_name} loaded from {self.cache_dir}")
    
    def get_model_info(self, model_name: str = "lr_model") -> Dict:
        """Get model information."""
        info = {
            'metadata': self.model_metadata.get(model_name, {}),
            'metrics': self.metrics.get(model_name, {}),
            'exists': model_name in self.models
        }
        return info
    
    def list_models(self) -> Dict:
        """List all available models."""
        return {
            'cached_models': [m.stem for m in self.cache_dir.glob('*.pkl') 
                            if '_metadata' not in m.stem],
            'loaded_models': list(self.models.keys()),
            'metrics': self.metrics
        }
    
    def clear_prediction_cache(self):
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
