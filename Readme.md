# 🖥️ VM Scheduler CloudSim

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive, AI/ML-driven Virtual Machine (VM) scheduling simulator designed to optimize resource allocation in cloud computing environments. By leveraging machine learning to predict CPU workloads, this system preemptively allocates resources, significantly improving upon traditional reactive heuristic-based scheduling methods.

---

## 📖 Problem Statement

In cloud computing environments, efficiently scheduling Virtual Machines (VMs) onto a limited number of servers is an exponentially complex, NP-hard problem. 

Traditional VM scheduling algorithms rely on static or heuristic-based approaches, which often result in suboptimal server states—either underloaded, overloaded, or unbalanced. These reactive methods fail to effectively anticipate the dynamic nature of incoming workloads, leading to inefficiencies such as increased energy consumption, longer processing times, and reduced system performance.

**The Solution:** This project addresses these challenges by using AI/ML techniques to predict CPU workloads in advance. By forecasting expected demand, the system can proactively optimize VM placement, reducing the need for reactive migrations, balancing energy consumption, and operating the cloud infrastructure at peak efficiency.

---

## ✨ Key Features

This project includes a fully interactive **Streamlit Dashboard** that allows users to monitor, train, predict, and allocate VMs dynamically. 

- **📈 Data Analysis**: Upload real-world CPU usage datasets (CSV/TXT) or generate synthetic data. Automatically identify peak usage hours and compute hourly statistics.
- **🤖 Model Training**: Train predictive models (Linear Regression, GRU, Bidirectional LSTM) on historical CPU data. Linear Regression is used as the primary fast-inference model.
- **⚡ Predictions & Allocation**: Generate multi-step future CPU predictions and run the `VMAllocator` to preemptively map VMs to hosts.
- **📊 Real-Time Metrics**: Track simulated Datacenter performance index, health status, average CPU/Memory/Storage usage, and system alerts.
- **ℹ️ System Info**: View detailed host-level resource utilization and active model parameters.

---

## 📁 Project Structure

```text
VM_Scheduler-CloudSim/
├── app/                      # Core application logic & Streamlit GUI
│   ├── gui_app.py            # Main Streamlit dashboard script
│   ├── run_gui.py            # Quick-start launcher script
│   ├── data_processor.py     # Data loading, synthesis, and feature engineering
│   ├── model_manager.py      # ML model training, evaluation, and prediction
│   ├── vm_allocator.py       # Cloud datacenter and VM allocation logic
│   └── metrics_tracker.py    # System performance monitoring
├── data/                     # Datasets
├── docs/                     # Project documentation
├── Phase 1 to 4/             # Jupyter notebooks containing research & model comparisons
├── model_cache/              # Serialized trained models
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
└── Readme.md                 # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Local Setup

**Prerequisites:** Python 3.8 or higher.

Clone the repository and install dependencies:
```bash
git clone https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh.git
cd VM_Scheduler-CloudSwaleh

# Install required packages
pip install -r requirements.txt
```

**Run the Dashboard:**
You can start the interactive Streamlit dashboard using the quick-start script:
```bash
python app/run_gui.py
```
*Alternatively, run Streamlit directly:*
```bash
streamlit run app/gui_app.py
```
The dashboard will open in your default browser at `http://localhost:8501/`.

### 2. Docker Setup

You can containerize and run the application using Docker:

```bash
# Build the Docker image
docker build -t vm-schedular .

# Run the container
docker run -p 8501:8501 vm-schedular
```
Open your browser and navigate to `http://localhost:8501/`.

---

## 🧠 Machine Learning Models

During the research phases, three models were evaluated for CPU usage prediction:

1. **Linear Regression**: Selected as the primary model. R² Score ~0.62. Fast training, comparable accuracy, and easy integration.
2. **GRU (Gated Recurrent Unit)**: Captures temporal dependencies but requires longer training with minimal accuracy improvement. R² Score ~0.62.
3. **Bidirectional LSTM**: Handles complex patterns but comes with high computational cost. R² Score ~0.61.

---

## 📊 Evaluation & Results

The AI/ML-based predictive scheduling approach was benchmarked against traditional Heuristic-based scheduling using simulated environments (`SimPy`).

**AI/ML-Based Approach:**
- **Resource Utilization**: Achieved **10–15% improvement**.
- **Energy Efficiency**: Reduced energy consumption by **8–12%** compared to reactive methods.
- **Scheduling Overhead**: Significantly lowered due to preemptive resource allocation, avoiding costly, reactive VM migrations.

**Heuristic-Based Approach:**
- Struggled to adapt to sudden demand spikes.
- Exhibited higher energy consumption, reactive migration costs, and processing delays.

---

## 📄 License

This project is licensed under the MIT License. See the `MIT LICENSE` file for details.