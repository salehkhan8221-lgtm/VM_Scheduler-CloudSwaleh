"""
Optimized Data Processing Module for VM Scheduler CloudSim
Handles efficient data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from functools import lru_cache
from typing import Tuple, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Efficient data processing with caching and memory optimization."""
    
    def __init__(self, dataset_path: str = None):
        """Initialize data processor with optional dataset path."""
        self.dataset_path = dataset_path
        self._cache = {}
        self.df = None
        self.stats = {}
        self.last_synthetic_seed = None
    
    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset efficiently with memory optimization.
        
        Args:
            file_path: Path to CSV or TXT file
            
        Returns:
            Loaded dataframe
        """
        if file_path is None:
            file_path = self.dataset_path
        
        if file_path is None:
            raise ValueError("No file path provided")
        
        logger.info(f"Loading dataset from {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, dtype={'CPUUsage': 'float32'})
            else:
                # For TXT files - assume single column format
                data = np.loadtxt(file_path, dtype='float32')
                self.df = pd.DataFrame(data, columns=['CPUUsage'])
            
            # Memory optimization
            self._optimize_memory()
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _optimize_memory(self):
        """Optimize dataframe memory usage."""
        for col in self.df.columns:
            if self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')
            elif self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype('int32')
    
    def standardize_columns(self):
        """Standardize column names."""
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        self.df.columns = [col.strip().replace(' ', '_') for col in self.df.columns]
        logger.info(f"Standardized columns: {list(self.df.columns)}")
        return self
    
    def format_timestamps(self, timestamp_col: str = 'Timestamp', 
                         format_str: str = '%d-%m-%Y %H:%M') -> 'DataProcessor':
        """
        Convert timestamp column to datetime format.
        
        Args:
            timestamp_col: Column name containing timestamps
            format_str: Datetime format string
            
        Returns:
            Self for method chaining
        """
        if timestamp_col in self.df.columns:
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col], format=format_str)
            logger.info(f"Formatted timestamps in column: {timestamp_col}")
        
        return self
    
    def extract_time_features(self) -> 'DataProcessor':
        """
        Extract temporal features from timestamp column.
        
        Returns:
            Self for method chaining
        """
        if 'Timestamp' in self.df.columns:
            self.df['Hour'] = self.df['Timestamp'].dt.hour
            self.df['DayOfWeek'] = self.df['Timestamp'].dt.dayofweek
            self.df['Month'] = self.df['Timestamp'].dt.month
            logger.info("Extracted temporal features")
        
        return self
    
    def calculate_statistics(self) -> Dict:
        """Calculate and cache CPU usage statistics."""
        if self.df is None or len(self.df) == 0:
            return {}
        
        cpu_col = self._find_cpu_column()
        
        self.stats = {
            'mean': self.df[cpu_col].mean(),
            'std': self.df[cpu_col].std(),
            'min': self.df[cpu_col].min(),
            'max': self.df[cpu_col].max(),
            'median': self.df[cpu_col].median(),
            'count': len(self.df)
        }
        
        logger.info(f"Statistics calculated: Mean={self.stats['mean']:.2f}, "
                   f"Std={self.stats['std']:.2f}, Max={self.stats['max']:.2f}")
        
        return self.stats
    
    def get_hourly_statistics(self) -> pd.DataFrame:
        """Get hourly average CPU usage efficiently."""
        if self.df is None or 'Hour' not in self.df.columns:
            raise ValueError("Hour column not found. Run extract_time_features() first.")
        
        cpu_col = self._find_cpu_column()
        hourly_stats = self.df.groupby('Hour')[cpu_col].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        logger.info("Hourly statistics computed")
        return hourly_stats
    
    def find_peak_hour(self) -> Tuple[int, float]:
        """Find peak hour with highest average CPU usage."""
        hourly_stats = self.get_hourly_statistics()
        peak_hour = hourly_stats['mean'].idxmax()
        peak_value = hourly_stats['mean'].max()
        
        logger.info(f"Peak hour: {peak_hour} with average CPU usage: {peak_value:.2f}%")
        return peak_hour, peak_value
    
    def normalize_data(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Normalize data to 0-1 range.
        
        Args:
            columns: Specific columns to normalize
            
        Returns:
            Normalized data as numpy array
        """
        if columns is None:
            cpu_col = self._find_cpu_column()
            columns = [cpu_col]
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self._cache['scaler'] = scaler
        
        normalized = scaler.fit_transform(self.df[columns])
        logger.info(f"Data normalized using MinMaxScaler")
        
        return normalized
    
    def create_sequences(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        cpu_col = self._find_cpu_column()
        data = self.df[cpu_col].values
        
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
        
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")
        return np.array(sequences, dtype='float32'), np.array(targets, dtype='float32')
    
    def remove_outliers(self, threshold: float = 3.0) -> 'DataProcessor':
        """
        Remove outliers using z-score method.
        
        Args:
            threshold: Z-score threshold
            
        Returns:
            Self for method chaining
        """
        cpu_col = self._find_cpu_column()
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(self.df[cpu_col]))
        initial_shape = self.df.shape[0]
        
        self.df = self.df[z_scores < threshold]
        
        removed = initial_shape - self.df.shape[0]
        logger.info(f"Removed {removed} outliers from {initial_shape} rows")
        
        return self
    
    def _find_cpu_column(self) -> str:
        """Find CPU usage column name."""
        cpu_variants = ['CPUUsage', 'CPU_Usage', 'cpu_usage', 'CPU', 'cpu']
        
        for col in self.df.columns:
            if col in cpu_variants:
                return col
        
        # If not found, assume first numeric column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError("No CPU usage column found in dataframe")
    
    def get_sample(self, size: int = 1000) -> pd.DataFrame:
        """Get random sample of data."""
        return self.df.sample(min(size, len(self.df)))
    
    def generate_synthetic_data(self, num_samples: int = 5000, seed: Optional[int] = None,
                                 pattern: str = 'mixed') -> pd.DataFrame:
        """
        Generate varied synthetic CPU usage data.
        
        Args:
            num_samples: Number of data points to generate
            seed: Random seed for reproducibility (None = random each time)
            pattern: Type of pattern - 'sine', 'noise', 'mixed', 'workload'
            
        Returns:
            DataFrame with timestamp and CPU usage
        """
        if seed is not None:
            np.random.seed(seed)
            self.last_synthetic_seed = seed
        else:
            # Random seed each time for varied data
            random_seed = np.random.randint(0, 10000)
            np.random.seed(random_seed)
            self.last_synthetic_seed = random_seed
        
        timestamps = pd.date_range('2023-01-01', periods=num_samples, freq='15min')
        
        if pattern == 'sine':
            # Pure sine wave
            cpu_usage = 50 + 30 * np.sin(np.arange(num_samples) / 100) + np.random.normal(0, 3, num_samples)
        
        elif pattern == 'noise':
            # Pure noise with baseline
            cpu_usage = 50 + np.random.normal(0, 15, num_samples)
        
        elif pattern == 'workload':
            # Multiple workload patterns
            base = 40
            trend = np.linspace(0, 30, num_samples) * np.sin(np.arange(num_samples) / 200)
            hourly = 20 * np.sin(2 * np.pi * (np.arange(num_samples) % 96) / 96)  # 24-hour cycle
            noise = np.random.normal(0, 5, num_samples)
            cpu_usage = base + trend + hourly + noise
        
        else:  # 'mixed' - most realistic
            # Combination of patterns
            base = 45
            
            # Trend component
            trend = np.linspace(0, 15, num_samples) * np.sin(np.arange(num_samples) / 300)
            
            # Daily pattern (24-hour cycle)
            daily = 15 * np.sin(2 * np.pi * (np.arange(num_samples) % 96) / 96)
            
            # Weekly pattern
            weekly = 8 * np.sin(2 * np.pi * (np.arange(num_samples) % 672) / 672)
            
            # Random spikes (workload bursts)
            spikes = np.random.poisson(0.1, num_samples) * np.random.uniform(10, 30, num_samples)
            
            # Noise
            noise = np.random.normal(0, 4, num_samples)
            
            cpu_usage = base + trend + daily + weekly + spikes + noise
        
        # Clip to valid range
        cpu_usage = np.clip(cpu_usage, 5, 100)
        
        self.df = pd.DataFrame({
            'Timestamp': timestamps,
            'CPUUsage': cpu_usage.astype('float32')
        })
        
        logger.info(f"Synthetic data generated: {self.df.shape} samples (seed={self.last_synthetic_seed}, pattern={pattern})")
        return self.df
    
    def reset(self):
        """Reset processor state."""
        self.df = None
        self._cache = {}
        self.stats = {}
        logger.info("DataProcessor reset")
