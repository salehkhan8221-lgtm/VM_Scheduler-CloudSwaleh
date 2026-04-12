# VM Scheduler CloudSim - Enhanced Version

## Overview

This enhanced version of the VM Scheduler CloudSim project includes:

✅ **Optimized Performance**
- Efficient data loading with memory optimization
- Cached model predictions
- Batch processing capabilities
- Real-time metrics tracking

✅ **Advanced Features**
- Multiple VM allocation strategies
- Resource constraint management
- Real-time monitoring and alerts
- Performance analytics

✅ **Interactive GUI Dashboard** (NEW!)
- Web-based Streamlit interface
- Live data analysis and visualization
- Model training and evaluation
- Prediction and allocation simulation
- Real-time metrics monitoring

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

#### Option 1: Automated Setup (Recommended)

```bash
# Navigate to project directory
cd "c:\Users\Administrator\Desktop\VM_Scheduler-CloudSim"

# Install dependencies automatically
python run_gui.py
```

This will:
1. Verify Python version
2. Install all required packages
3. Launch the Streamlit dashboard automatically

#### Option 2: Manual Setup

```bash
# Navigate to project directory
cd "c:\Users\Administrator\Desktop\VM_Scheduler-CloudSim"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run gui_app.py
```

### Accessing the Dashboard

Once started, the dashboard automatically opens at:
```
http://localhost:8501
```

Or manually navigate to it in your browser.

---

## 📊 GUI Dashboard Features

### Tab 1: Data Analysis 📈
- **Load Dataset**: Upload or use default dataset
- **Hourly Statistics**: View CPU usage patterns by hour
- **Peak Hour Analysis**: Identify peak usage times
- **Statistical Metrics**: Mean, std dev, min, max values

**Key Functions:**
- Load CSV/TXT files
- Automatic data preprocessing
- Time-based feature extraction
- Statistical calculations

### Tab 2: Model Training 🤖
- **Train Linear Regression**: Build ML model
- **Model Evaluation**: View performance metrics (R², RMSE, MAE)
- **Model Comparison**: Compare different model performances
- **Save Models**: Cache trained models

**Metrics Calculated:**
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)

### Tab 3: Predictions & Allocation ⚡
- **Generate Predictions**: Forecast next N CPU values
- **VM Allocation**: Allocate VMs based on predictions
- **Allocation Report**: Success rates and allocation details
- **Visualization**: Prediction trends

**Allocation Strategies:**
- Threshold-based allocation
- Load balancing
- Predictive allocation

### Tab 4: Metrics & Monitoring 📊
- **Real-time Metrics**: Track system performance
- **Performance Index**: Overall system health score
- **Alert System**: Automatic alerting on thresholds
- **Trends Analysis**: Identify patterns over time
- **Export Data**: Save metrics to CSV

**Monitored Metrics:**
- CPU, Memory, Storage utilization
- VM count and distribution
- Host availability
- Allocation success rate

### Tab 5: System Information ℹ️
- **Datacenter Status**: Overall capacity and utilization
- **Host Details**: Per-host resource allocation
- **Model Information**: Trained model metadata
- **Resource Visualization**: Host utilization charts

---

## 🔧 Core Modules

### 1. `data_processor.py`
Efficient data handling with memory optimization.

```python
from data_processor import DataProcessor

processor = DataProcessor()
df = processor.load_dataset('data.csv')
processor.standardize_columns()
processor.extract_time_features()
stats = processor.calculate_statistics()
```

**Key Methods:**
- `load_dataset()` - Load CSV/TXT files
- `normalize_data()` - MinMax scaling
- `create_sequences()` - Time series preparation
- `calculate_statistics()` - Statistical analysis
- `get_hourly_statistics()` - Hourly aggregation

### 2. `model_manager.py`
ML model management with caching.

```python
from model_manager import ModelManager

manager = ModelManager()
manager.prepare_data(X, y)
manager.train_linear_regression(X_train, y_train)
metrics = manager.evaluate(X_test, y_test)
predictions = manager.predict_next_values(last_sequence)
manager.save_model()
```

**Key Methods:**
- `train_linear_regression()` - Train LR model
- `prepare_data()` - Normalize and split data
- `predict()` - Make predictions
- `predict_next_values()` - Multi-step forecasting
- `evaluate()` - Calculate performance metrics
- `save_model()` / `load_model()` - Persistence

### 3. `vm_allocator.py`
Advanced VM allocation with resource management.

```python
from vm_allocator import VMAllocator

allocator = VMAllocator(num_hosts=10)
allocator.set_thresholds(cpu=80, memory=85)

# Create and allocate VMs
vm = allocator.create_vm(cpu=1.0, memory=1024)
success, host_id, msg = allocator.allocate_load_balanced(vm)

# Get statistics
stats = allocator.get_host_statistics()
util = allocator.get_datacenter_utilization()
```

**Key Methods:**
- `allocate_threshold_based()` - CPU threshold allocation
- `allocate_load_balanced()` - Distribute load evenly
- `allocate_predictive()` - Allocation based on forecasts
- `get_datacenter_utilization()` - Overall metrics
- `get_host_statistics()` - Per-host details

### 4. `metrics_tracker.py`
Real-time metrics and performance monitoring.

```python
from metrics_tracker import MetricsTracker

tracker = MetricsTracker()
tracker.record_metric(
    timestamp=0,
    cpu_usage=75.5,
    memory_usage=60.2,
    vms_count=5
)

summary = tracker.get_summary()
alerts = tracker.get_alerts()
health = tracker.get_health_status()
```

**Key Methods:**
- `record_metric()` - Log performance data
- `get_summary()` - Aggregate statistics
- `get_trend_analysis()` - Trend detection
- `export_metrics()` - Export to CSV
- `get_performance_index()` - Health score
- `get_health_status()` - System status

---

## 📈 Performance Improvements

### 1. Memory Optimization
- Converts float64 to float32 automatically
- Efficient data structure usage
- Deque-based circular buffer for history

### 2. Computation Efficiency
- Cached model predictions
- Batch processing support
- Vectorized NumPy operations

### 3. Fast Data Loading
- Optimized CSV parsing
- Stream processing for large files
- MinMax scaling optimization

### 4. Scalable Architecture
- Modular design for easy extensions
- Support for multiple models
- Parallel processing capability

---

## 🎯 Usage Examples

### Example 1: Complete Data Analysis Pipeline

```python
from data_processor import DataProcessor

processor = DataProcessor()

# Load and process
df = processor.load_dataset('dataset.csv')
processor.standardize_columns()
processor.format_timestamps()
processor.extract_time_features()

# Get statistics
stats = processor.calculate_statistics()
print(f"Peak Hour: {processor.find_peak_hour()}")

# Visualize
hourly_stats = processor.get_hourly_statistics()
print(hourly_stats)
```

### Example 2: Model Training and Prediction

```python
from data_processor import DataProcessor
from model_manager import ModelManager

# Prepare data
processor = DataProcessor()
df = processor.load_dataset('dataset.csv')
X, y = processor.create_sequences(sequence_length=10)

# Train model
manager = ModelManager()
X_train, X_test, y_train, y_test = manager.prepare_data(X, y)
manager.train_linear_regression(X_train, y_train)

# Evaluate
metrics = manager.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2']:.4f}")

# Predict future values
last_sequence = X_train[-1]
future_predictions = manager.predict_next_values(last_sequence, num_steps=10)
print(f"Next 10 predictions: {future_predictions}")
```

### Example 3: VM Allocation with Resource Management

```python
from vm_allocator import VMAllocator
import numpy as np

# Setup datacenter
allocator = VMAllocator(num_hosts=10, cpu_per_host=8.0)
allocator.set_thresholds(cpu=80, memory=85)

# Simulate predictions and allocate
predicted_values = np.random.uniform(40, 95, 5)
allocation = allocator.allocate_predictive(predicted_values)

print(f"Success Rate: {allocation['successful']}/{allocation['total_vms']}")

# Get utilization
util = allocator.get_datacenter_utilization()
print(f"CPU Utilization: {util['cpu_utilization']:.2f}%")
print(f"Total VMs: {util['total_vms']}")
```

### Example 4: Real-time Monitoring

```python
from metrics_tracker import MetricsTracker

tracker = MetricsTracker()

# Record metrics over time
for i in range(100):
    cpu = 50 + 30 * np.sin(i / 10)
    tracker.record_metric(
        timestamp=i,
        cpu_usage=cpu,
        memory_usage=60,
        vms_count=i // 10
    )

# Analyze
summary = tracker.get_summary()
print(f"Performance Index: {tracker.get_performance_index()}/100")
print(f"Health Status: {tracker.get_health_status()}")

# Get alerts
alerts = tracker.get_alerts()
print(f"Total alerts: {len(alerts)}")
```

---

## 🔍 Configuration

### Datacenter Configuration (in GUI)
- Number of hosts (5-50)
- CPU cores per host (2-16)
- Memory per host (1GB-64GB)
- Storage per host

### Allocation Thresholds
- CPU threshold: 50-95%
- Memory threshold: 50-95%
- Storage threshold: 50-95%

### Model Configuration
- Sequence length: 5-30
- Test/Train split: 10-50%
- Prediction steps: 5-50

---

## 📊 Sample Workflow

1. **Load Data**
   - Upload CSV/TXT file via GUI
   - View data statistics and peak hours

2. **Train Model**
   - Configure sequence length and test size
   - Train Linear Regression model
   - Evaluate metrics

3. **Generate Predictions**
   - Click "Generate Predictions"
   - View forecast for next N timesteps

4. **Allocate VMs**
   - Based on predictions
   - View allocation success rate
   - Monitor resource utilization

5. **Monitor System**
   - Track real-time metrics
   - View alerts and warnings
   - Export data for analysis

---

## 🐛 Troubleshooting

### Dashboard Won't Start
```bash
# Ensure Streamlit is installed
pip install streamlit

# Try running directly
streamlit run gui_app.py
```

### Import Errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Dataset Not Found
- Place CSV files in the project directory
- Update file path in the GUI
- Check file format (must be CSV or TXT)

### Memory Issues
- Reduce dataset size
- Lower the max_history in MetricsTracker
- Use data sampling

---

## 📚 Additional Resources

### Files Structure
```
VM_Scheduler-CloudSim/
├── gui_app.py              # Main Streamlit dashboard
├── run_gui.py              # Launcher script
├── data_processor.py       # Data handling module
├── model_manager.py        # ML model management
├── vm_allocator.py         # VM allocation logic
├── metrics_tracker.py      # Real-time monitoring
├── requirements.txt        # Python dependencies
└── Dataset/
    ├── dataset.csv
    └── Dataset with timestamp.csv
```

### Default Dataset Location
```
Dataset/Dataset with timestamp.csv
```

---

## 🤝 Contributing

To extend functionality:

1. **Add New Allocation Strategy**: 
   - Modify `vm_allocator.py`
   - Implement new method in `VMAllocator` class

2. **Add New ML Model**:
   - Extend `model_manager.py`
   - Add training and prediction methods

3. **Add Dashboard Features**:
   - Update `gui_app.py`
   - Add new tabs or sections

---

## 📝 License

MIT License (as per project)

---

## 🚀 Performance Benchmarks

Expected performance with default settings:
- **Data Loading**: < 1 second (10K rows)
- **Model Training**: 1-5 seconds
- **Prediction**: < 100ms
- **VM Allocation**: < 50ms
- **Memory Usage**: < 500MB (with typical dataset)

---

## ❓ FAQ

**Q: Can I use my own dataset?**
A: Yes! Upload any CSV or TXT file with CPU usage data via the GUI.

**Q: What's the recommended number of hosts?**
A: Typically 5-20 hosts for datacenter simulation. Adjust based on your needs.

**Q: Can models be saved and loaded?**
A: Yes! Models are automatically cached and can be reloaded.

**Q: How do I export results?**
A: Use the "Export Metrics" feature in the Metrics tab.

**Q: Is it production-ready?**
A: The code is optimized for research and simulation. For production, add authentication, database persistence, and additional validation.

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example code
3. Check module docstrings
4. Verify all dependencies are installed

---

**Happy Scheduling! 🖥️**
