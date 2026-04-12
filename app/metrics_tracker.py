"""
Real-time Metrics Tracking Module
Tracks and aggregates system and allocation metrics over time.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track system and allocation metrics in real-time."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of history entries to keep
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.aggregates = {}
        self.alerts = []
        self.start_time = datetime.now()
    
    def record_metric(self, timestamp: int, cpu_usage: float,
                     memory_usage: float = 0, storage_usage: float = 0,
                     vms_count: int = 0, active_hosts: int = 0,
                     prediction: Optional[float] = None,
                     allocation_success: Optional[bool] = None) -> Dict:
        """
        Record a metric snapshot.
        
        Args:
            timestamp: Simulation timestamp
            cpu_usage: CPU utilization percentage
            memory_usage: Memory utilization percentage
            storage_usage: Storage utilization percentage
            vms_count: Number of active VMs
            active_hosts: Number of active hosts
            prediction: Predicted CPU value (optional)
            allocation_success: Whether allocation was successful (optional)
            
        Returns:
            The recorded metric dictionary
        """
        metric = {
            'timestamp': timestamp,
            'datetime': datetime.now(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'storage_usage': storage_usage,
            'vms_count': vms_count,
            'active_hosts': active_hosts,
            'prediction': prediction,
            'allocation_success': allocation_success,
            'avg_utilization': (cpu_usage + memory_usage + storage_usage) / 3
        }
        
        self.history.append(metric)
        
        # Check for alerts
        self._check_alerts(metric)
        
        return metric
    
    def _check_alerts(self, metric: Dict):
        """Check for alert conditions."""
        alerts_triggered = []
        
        if metric['cpu_usage'] > 90:
            alerts_triggered.append(f"High CPU usage: {metric['cpu_usage']:.2f}%")
        
        if metric['memory_usage'] > 90:
            alerts_triggered.append(f"High memory usage: {metric['memory_usage']:.2f}%")
        
        if metric['storage_usage'] > 90:
            alerts_triggered.append(f"High storage usage: {metric['storage_usage']:.2f}%")
        
        if metric['allocation_success'] is False:
            alerts_triggered.append("VM allocation failed")
        
        for alert in alerts_triggered:
            self.alerts.append({
                'timestamp': metric['timestamp'],
                'datetime': metric['datetime'],
                'message': alert
            })
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all recorded metrics."""
        if not self.history:
            return {}
        
        df = pd.DataFrame(list(self.history))
        
        summary = {
            'total_records': len(df),
            'time_period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'cpu': {
                'mean': df['cpu_usage'].mean(),
                'max': df['cpu_usage'].max(),
                'min': df['cpu_usage'].min(),
                'std': df['cpu_usage'].std()
            },
            'memory': {
                'mean': df['memory_usage'].mean(),
                'max': df['memory_usage'].max(),
                'min': df['memory_usage'].min(),
                'std': df['memory_usage'].std()
            },
            'storage': {
                'mean': df['storage_usage'].mean(),
                'max': df['storage_usage'].max(),
                'min': df['storage_usage'].min(),
                'std': df['storage_usage'].std()
            },
            'vms': {
                'mean': df['vms_count'].mean(),
                'max': df['vms_count'].max(),
                'min': df['vms_count'].min()
            },
            'avg_utilization': {
                'mean': df['avg_utilization'].mean(),
                'max': df['avg_utilization'].max(),
                'min': df['avg_utilization'].min()
            }
        }
        
        # Prediction accuracy if available
        if 'prediction' in df.columns and df['cpu_usage'] is not None:
            predictions = df[df['prediction'].notna()]
            if len(predictions) > 0:
                mae = np.abs(predictions['prediction'] - predictions['cpu_usage']).mean()
                summary['prediction_mae'] = mae
        
        # Allocation success rate
        if 'allocation_success' in df.columns:
            success_rate = df['allocation_success'].sum() / df['allocation_success'].notna().sum() * 100
            summary['allocation_success_rate'] = success_rate
        
        return summary
    
    def get_recent_metrics(self, last_n: int = 10) -> pd.DataFrame:
        """Get last N metrics as dataframe."""
        if not self.history:
            return pd.DataFrame()
        
        recent = list(self.history)[-last_n:]
        return pd.DataFrame(recent)
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]
    
    def get_statistics_by_hour(self) -> Optional[pd.DataFrame]:
        """Get hourly statistics if timestamps are available."""
        if not self.history:
            return None
        
        df = pd.DataFrame(list(self.history))
        
        if 'datetime' not in df.columns:
            return None
        
        df['hour'] = df['datetime'].dt.floor('H')
        
        hourly_stats = df.groupby('hour').agg({
            'cpu_usage': ['mean', 'max', 'min'],
            'memory_usage': ['mean', 'max'],
            'vms_count': ['mean', 'max'],
            'avg_utilization': 'mean'
        }).round(2)
        
        return hourly_stats
    
    def get_trend_analysis(self, metric: str = 'cpu_usage', window: int = 10) -> Dict:
        """
        Analyze trend of a specific metric.
        
        Args:
            metric: Metric name to analyze
            window: Moving average window
            
        Returns:
            Dictionary with trend information
        """
        if not self.history or len(self.history) < window:
            return {}
        
        df = pd.DataFrame(list(self.history))
        
        if metric not in df.columns:
            return {}
        
        values = df[metric].values
        
        # Calculate moving average
        moving_avg = pd.Series(values).rolling(window).mean()
        
        # Trend direction (increasing/decreasing)
        recent_ma = moving_avg[-5:].mean()
        prev_ma = moving_avg[-10:-5].mean()
        
        trend = "increasing" if recent_ma > prev_ma else "decreasing"
        
        return {
            'metric': metric,
            'current': values[-1],
            'mean': values.mean(),
            'max': values.max(),
            'min': values.min(),
            'std': values.std(),
            'trend': trend,
            'trend_strength': abs(recent_ma - prev_ma) / prev_ma * 100 if prev_ma > 0 else 0
        }
    
    def export_metrics(self, filepath: str = 'metrics.csv'):
        """Export metrics to CSV file."""
        if not self.history:
            logger.warning("No metrics to export")
            return
        
        df = pd.DataFrame(list(self.history))
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics exported to {filepath}")
    
    def export_alerts(self, filepath: str = 'alerts.csv'):
        """Export alerts to CSV file."""
        if not self.alerts:
            logger.warning("No alerts to export")
            return
        
        df = pd.DataFrame(self.alerts)
        df.to_csv(filepath, index=False)
        logger.info(f"Alerts exported to {filepath}")
    
    def reset(self):
        """Reset tracker."""
        self.history.clear()
        self.alerts.clear()
        self.aggregates.clear()
        logger.info("Metrics tracker reset")
    
    def get_performance_index(self) -> float:
        """
        Calculate overall performance index (0-100).
        Higher is better.
        """
        summary = self.get_summary()
        
        if not summary:
            return 0
        
        # Normalize metrics inversely (lower usage = better)
        cpu_score = max(0, 100 - summary['cpu']['mean'])
        memory_score = max(0, 100 - summary['memory']['mean'])
        storage_score = max(0, 100 - summary['storage']['mean'])
        
        # Weighted average (CPU has highest weight)
        performance_index = (cpu_score * 0.5 + memory_score * 0.3 + storage_score * 0.2)
        
        return round(performance_index, 2)
    
    def get_health_status(self) -> str:
        """Get health status based on metrics."""
        if not self.history:
            return "No data"
        
        summary = self.get_summary()
        
        avg_cpu = summary['cpu']['mean']
        avg_memory = summary['memory']['mean']
        
        if avg_cpu > 80 or avg_memory > 80:
            return "Critical"
        elif avg_cpu > 60 or avg_memory > 60:
            return "Warning"
        else:
            return "Healthy"
