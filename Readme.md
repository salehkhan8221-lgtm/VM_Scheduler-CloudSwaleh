# AI-Powered VM Scheduler for Cloud Computing

> **Predictive workload forecasting for proactive Virtual Machine (VM) scheduling and resource optimization in cloud environments.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![CloudSim](https://img.shields.io/badge/CloudSim-Simulation-orange)](https://www.cloudbus.org/cloudsim/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Time--Series%20Forecasting-green)](https://scikit-learn.org/)

## Overview

Cloud infrastructure must continuously decide **which VMs should run on which physical servers** while balancing resource utilization, energy consumption, scheduling overhead, and response time.

With **M VMs and N servers (M >> N)**, the number of possible VM-to-server configurations grows to **M^N**, making optimal scheduling computationally challenging. Traditional heuristic approaches are often reactive: they make placement and migration decisions using currently available information rather than anticipating future workload.

This project explores an **AI/ML-driven VM scheduling strategy** that predicts upcoming CPU utilization and uses those predictions to make more proactive scheduling decisions.

### Core Idea

```text
Historical CPU Usage
        │
        ▼
Data Preparation & Feature Engineering
        │
        ▼
CPU Workload Prediction
        │
        ▼
Predicted Future Demand
        │
        ▼
Proactive VM Placement / Migration
        │
        ▼
CloudSim Simulation
        │
        ▼
Performance Comparison
(AI/ML vs Heuristic)
```

The objective is to find a better balance between **resource utilization and system efficiency**, while reducing unnecessary reactive scheduling decisions.

---

## Project Goals

- Forecast future CPU workload using machine learning.
- Use workload predictions to improve VM placement decisions.
- Integrate the predictive model with a CloudSim-based scheduling framework.
- Compare predictive scheduling against traditional heuristic-based scheduling.
- Evaluate the system using resource utilization, energy consumption, scheduling overhead, and response time.

---

# Project Workflow

The project is organized into four phases.

## Phase 1 — Data Analysis & Insights

**Objective:** Understand CPU utilization patterns and prepare the data for predictive modeling.

[Open Phase 1 →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%201)

### Data Collection

- CPU utilization data was generated through **CloudSim simulations**.
- The original dataset contained **348 files**, each with **288 CPU usage observations**.
- The files were consolidated into a single dataset containing **100,000+ observations**.

### Data Preparation

- Standardized column names such as `Timestamp` and `CPU_Usage`.
- Converted timestamps to the `%d-%m-%Y %H:%M` datetime format.
- Extracted the **hour of day** as a feature for workload analysis.

### Analysis

The dataset was grouped by hour to calculate average CPU utilization and identify periods of relatively high and low workload.

📓 **Analysis notebook:**  
[CPU Usage Analysis](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/blob/main/Phase%201/CPU_usage_analysis.ipynb)

> **Note:** The source README currently contains placeholder values (`Hour X` and `Y%`) for the exact peak-hour result. The actual value should be added from the analysis notebook before publishing the final README.

---

## Phase 2 — Model Development & Evaluation

**Objective:** Build and compare models capable of forecasting future CPU utilization.

[Open Phase 2 →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%202)

### Preprocessing

- Consolidated the raw CPU utilization data.
- Removed unnecessary columns.
- Handled missing values.
- Created a `Next_CPU_Usage` target by shifting CPU usage values to represent future workload.

### Models Evaluated

| Model | R² Score | Key Characteristics |
|---|---:|---|
| **Linear Regression** | ~0.62 | Simple, fast, and comparatively efficient |
| **GRU** | ~0.62 | Captures temporal dependencies with lower complexity than LSTM |
| **Bidirectional LSTM** | ~0.61 | Captures complex sequential dependencies but has higher computational cost |

### Model Comparison

#### 1. Linear Regression

[View notebook →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/blob/main/Phase%202/Modeling/01_Linear%20Regression.ipynb)

**Strengths**
- Simple to implement.
- Fast training and inference.
- Comparable predictive performance to the more complex models tested.

**Limitation**
- Assumes a linear relationship and may not capture complex nonlinear workload patterns.

#### 2. GRU — Gated Recurrent Unit

[View notebook →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/blob/main/Phase%202/Modeling/02_GRU%20Model.ipynb)

**Strengths**
- Captures temporal dependencies.
- Less complex than LSTM.
- Suitable for sequential workload data.

**Limitations**
- Longer training time than Linear Regression.
- Provided minimal improvement in predictive accuracy in this experiment.

#### 3. Bidirectional LSTM

[View notebook →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/blob/main/Phase%202/Modeling/03_Bidirectional%20LSTM.ipynb)

**Strengths**
- Models complex sequential patterns.
- Captures dependencies across the sequence.

**Limitations**
- Higher computational cost.
- Slower training.
- Did not provide a significant accuracy improvement in this experiment.

### Model Selection

**Linear Regression** was selected for integration because it delivered comparable predictive performance while offering substantially lower complexity and faster training/inference.

---

## Phase 3 — Model Integration

**Objective:** Integrate the selected predictive model into the VM scheduling framework.

[Open Phase 3 →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%203)

### Framework

- **CloudSim** — cloud infrastructure simulation and VM scheduling.
- **Eclipse IDE** — development, integration, and debugging environment.
- **Linear Regression** — selected CPU workload prediction model.

### Integration Process

1. Export the trained Linear Regression model as a serialized object.
2. Integrate the model into the CloudSim simulation workflow.
3. Feed predicted CPU utilization into the VM allocation policy.
4. Adjust VM placement and migration decisions using anticipated workload.
5. Simulate changing workload conditions and observe system behavior.

### Result

The predictive model was integrated successfully into the VM scheduling process, with initial observations indicating more balanced resource utilization compared with the heuristic-based approach.

---

## Phase 4 — System Evaluation & Benchmarking

**Objective:** Evaluate the integrated system and compare predictive scheduling with traditional heuristic scheduling.

[Open Phase 4 →](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%204)

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Resource Utilization** | Percentage of available server resources actively utilized |
| **Energy Efficiency** | Total energy consumed during simulation |
| **Scheduling Overhead** | Computational/time cost of scheduling decisions |
| **Response Time** | Time required to handle incoming workloads |

### AI/ML-Based Scheduling

The predictive scheduler uses forecasted CPU utilization to make proactive allocation decisions and reduce dependence on reactive migrations.

### Heuristic-Based Scheduling

The baseline scheduler relies on static rules and existing workload information, making it more reactive to workload fluctuations.

### Reported Results

| Metric | Reported Improvement |
|---|---:|
| Resource Utilization | **10–15% improvement** |
| Energy Consumption | **8–12% reduction** |
| Scheduling Overhead | **Lower than heuristic approach** |

> These figures are reported in the current project documentation and should be interpreted in the context of the project's simulation setup and evaluation methodology.

---

# Technology Stack

| Category | Technology |
|---|---|
| Programming / Analysis | Python |
| Machine Learning | Linear Regression, GRU, Bidirectional LSTM |
| Cloud Simulation | CloudSim |
| Development | Eclipse IDE |
| Containerization | Docker |
| Data Processing | Pandas / Python data-processing workflow |
| Model Evaluation | MSE, R² |
| Interface | Streamlit |

---

# Repository Structure

```text
VM_Scheduler-CloudSwaleh/
│
├── Phase 1/
│   └── CPU_usage_analysis.ipynb
│
├── Phase 2/
│   └── Modeling/
│       ├── 01_Linear Regression.ipynb
│       ├── 02_GRU Model.ipynb
│       └── 03_Bidirectional LSTM.ipynb
│
├── Phase 3/
│   └── CloudSim integration
│
├── Phase 4/
│   └── Evaluation and comparison
│
├── app/
│   └── Core application logic
│
├── data/
│   └── Dataset
│
└── docs/
    └── Documentation
```

> The structure above reflects the project organization described in the current README. Add or adjust individual filenames if the repository contains additional files.

---

# Running the Application with Docker

### 1. Build the image

```bash
docker build -t vm-scheduler .
```

### 2. Start the container

```bash
docker run -p 8501:8501 vm-scheduler
```

### 3. Open the application

Open:

```text
http://localhost:8501/
```

If the application is running on another machine or server, replace `localhost` with that machine's accessible host/IP address and ensure port `8501` is exposed through the relevant network or firewall configuration.

---

# Key Takeaways

- VM scheduling is a computationally challenging optimization problem.
- Forecasting CPU utilization enables **proactive** rather than purely reactive scheduling.
- Three predictive approaches were evaluated: **Linear Regression, GRU, and Bidirectional LSTM**.
- Linear Regression was selected because its predictive performance was comparable while being significantly simpler and faster.
- The integrated predictive scheduling approach reported improvements in resource utilization and energy consumption relative to the heuristic baseline.
- CloudSim provides the simulation environment for evaluating scheduling behavior before deployment in real infrastructure.

---

# Future Improvements

Potential directions for extending the project include:

- Improving workload forecasting accuracy with richer temporal and system-level features.
- Evaluating additional forecasting and ensemble models.
- Performing broader hyperparameter tuning for recurrent models.
- Testing the scheduler under more diverse workload patterns.
- Conducting larger-scale experiments with different VM/server configurations.
- Adding automated experiment tracking and reproducible benchmarking.

---

## Project Links

- **Repository:** [VM_Scheduler-CloudSwaleh](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh)
- **Phase 1 — Data Analysis:** [View Phase 1](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%201)
- **Phase 2 — Model Development:** [View Phase 2](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%202)
- **Phase 3 — Integration:** [View Phase 3](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%203)
- **Phase 4 — Evaluation:** [View Phase 4](https://github.com/salehkhan8221-lgtm/VM_Scheduler-CloudSwaleh/tree/main/Phase%204)

---

## Author

**Swaleh Khan**

This project demonstrates the application of machine learning and cloud simulation to **predict workload demand and improve VM scheduling decisions**.
