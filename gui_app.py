"""
VM Scheduler CloudSim - Interactive Dashboard with Streamlit
Complete GUI application for monitoring, prediction, and allocation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_processor import DataProcessor
from model_manager import ModelManager
from vm_allocator import VMAllocator
from metrics_tracker import MetricsTracker

# ==================== STREAMLIT CONFIG ====================
st.set_page_config(
    page_title="VM Scheduler CloudSim",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.title("🔧 Configuration")

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'vm_allocator' not in st.session_state:
    st.session_state.vm_allocator = VMAllocator(num_hosts=10)
if 'metrics_tracker' not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker()

# Sidebar sections
with st.sidebar:
    st.header("📊 System Configuration")
    
    # Dataset selection
    st.subheader("1. Dataset")
    dataset_option = st.radio(
        "Select dataset:",
        ["Upload CSV/TXT", "Generate Synthetic Data"],
        key="dataset_option"
    )
    
    if dataset_option == "Upload CSV/TXT":
        uploaded_file = st.file_uploader("Upload dataset", type=['csv', 'txt'])
        if uploaded_file:
            try:
                with st.spinner("Loading dataset..."):
                    df = st.session_state.data_processor.load_dataset(uploaded_file.name)
                st.success("✅ Dataset loaded successfully!")
                st.write(f"Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    else:  # Generate Synthetic Data
        st.write("**Synthetic Data Generator**")
        num_samples = st.slider("Number of Samples", 1000, 20000, 5000, step=500)
        pattern = st.selectbox(
            "Data Pattern",
            ["mixed", "sine", "workload", "noise"],
            help="mixed: Realistic pattern | sine: Pure wave | workload: Workload spikes | noise: Random noise"
        )
        random_seed = st.checkbox("Use Random Seed (varied data)", value=True)
        
        if st.button("🔄 Generate New Data"):
            try:
                with st.spinner("Generating synthetic data..."):
                    seed = None if random_seed else 42
                    df = st.session_state.data_processor.generate_synthetic_data(
                        num_samples=num_samples,
                        seed=seed,
                        pattern=pattern
                    )
                st.success("✅ Synthetic data generated!")
                st.write(f"Shape: {df.shape} | Pattern: {pattern}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Model configuration
    st.subheader("2. ML Model")
    sequence_length = st.slider("Sequence Length", 5, 30, 10)
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    
    # VM Allocator configuration
    st.subheader("3. Datacenter Setup")
    num_hosts = st.slider("Number of Hosts", 5, 50, 10)
    cpu_per_host = st.slider("CPU Cores per Host", 2.0, 16.0, 8.0)
    memory_per_host = st.slider("Memory per Host (MB)", 1024, 65536, 16384, step=1024)
    
    # Thresholds
    st.subheader("4. Allocation Thresholds")
    cpu_threshold = st.slider("CPU Threshold (%)", 50, 95, 80)
    memory_threshold = st.slider("Memory Threshold (%)", 50, 95, 85)
    
    # Prediction steps
    st.subheader("5. Prediction")
    pred_steps = st.slider("Prediction Steps", 5, 50, 10)
    
    # Update configurations
    if st.button("⚙️ Apply Configuration", key="apply_config"):
        st.session_state.vm_allocator = VMAllocator(
            num_hosts=num_hosts,
            cpu_per_host=cpu_per_host,
            memory_per_host=memory_per_host
        )
        st.session_state.vm_allocator.set_thresholds(cpu_threshold, memory_threshold)
        st.success("✅ Configuration applied!")

# ==================== MAIN CONTENT ====================
st.title("🖥️ VM Scheduler CloudSim - Interactive Dashboard")
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data Analysis",
    "🤖 Model Training",
    "⚡ Predictions & Allocation",
    "📊 Metrics & Monitoring",
    "ℹ️ System Info"
])

# ==================== TAB 1: DATA ANALYSIS ====================
with tab1:
    st.header("Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Process Loaded Data"):
            processor = st.session_state.data_processor
            
            try:
                if processor.df is not None and len(processor.df) > 0:
                    with st.spinner("Processing data..."):
                        processor.standardize_columns()
                        processor.format_timestamps()
                        processor.extract_time_features()
                        stats = processor.calculate_statistics()
                        
                        st.success("✅ Data processed successfully!")
                        
                        col1_1, col2_1 = st.columns(2)
                        with col1_1:
                            st.metric("Mean CPU Usage", f"{stats['mean']:.2f}%")
                            st.metric("Std Dev", f"{stats['std']:.2f}%")
                        with col2_1:
                            st.metric("Max CPU Usage", f"{stats['max']:.2f}%")
                            st.metric("Min CPU Usage", f"{stats['min']:.2f}%")
                else:
                    st.warning("No data loaded. Upload or generate data in the sidebar first.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("🎯 Find Peak Hour"):
            processor = st.session_state.data_processor
            
            try:
                if processor.df is not None:
                    peak_hour, peak_value = processor.find_peak_hour()
                    
                    st.success(f"✅ Peak Hour Found")
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Peak Hour", f"{peak_hour}:00")
                    with col2_2:
                        st.metric("Peak CPU Usage", f"{peak_value:.2f}%")
                else:
                    st.warning("Please load data first")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display hourly statistics
    st.subheader("Hourly CPU Usage Statistics")
    
    if st.button("📊 Show Hourly Stats"):
        processor = st.session_state.data_processor
        
        try:
            if processor.df is not None:
                hourly_stats = processor.get_hourly_statistics()
                
                # Display table
                st.dataframe(hourly_stats, use_container_width=True)
                
                # Create visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_stats.index,
                    y=hourly_stats['mean'],
                    mode='lines+markers',
                    name='Average CPU',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hourly_stats.index,
                    y=hourly_stats['max'],
                    mode='lines',
                    name='Max CPU',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Hourly CPU Usage Patterns",
                    xaxis_title="Hour of Day",
                    yaxis_title="CPU Usage (%)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please load data first")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== TAB 2: MODEL TRAINING ====================
with tab2:
    st.header("ML Model Training & Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train Linear Regression Model"):
            processor = st.session_state.data_processor
            model_manager = st.session_state.model_manager
            
            try:
                if processor.df is not None:
                    with st.spinner("Training model..."):
                        # Prepare data
                        X, y = processor.create_sequences(sequence_length)
                        X_scaled = processor.normalize_data()
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = model_manager.prepare_data(
                            X, y, test_size=test_size
                        )
                        
                        # Train model
                        model_manager.train_linear_regression(X_train, y_train)
                        
                        # Evaluate
                        metrics = model_manager.evaluate(X_test, y_test)
                        
                        st.success("✅ Model trained successfully!")
                        
                        col1_1, col2_1, col3_1 = st.columns(3)
                        with col1_1:
                            st.metric("R² Score", f"{metrics['r2']:.4f}")
                        with col2_1:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col3_1:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        
                        # Save model
                        model_manager.save_model()
                        st.info("Model saved to cache")
                else:
                    st.warning("Please load data first")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("💾 Save Trained Model"):
            model_manager = st.session_state.model_manager
            
            try:
                model_manager.save_model()
                st.success("✅ Model saved successfully!")
                
                # Show model info
                info = model_manager.get_model_info()
                st.json(info)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Model comparison
    st.subheader("Model Metrics Comparison")
    
    model_manager = st.session_state.model_manager
    
    if model_manager.metrics:
        metrics_df = pd.DataFrame([
            {"Model": name, **metrics}
            for name, metrics in model_manager.metrics.items()
        ])
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        for col in ['r2', 'rmse', 'mae']:
            if col in metrics_df.columns:
                fig.add_trace(go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df[col],
                    name=col.upper()
                ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train a model to see metrics")

# ==================== TAB 3: PREDICTIONS & ALLOCATION ====================
with tab3:
    st.header("CPU Prediction & VM Allocation")
    
    # Prediction section
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Generate Predictions"):
            processor = st.session_state.data_processor
            model_manager = st.session_state.model_manager
            
            try:
                if processor.df is not None and "lr_model" in model_manager.models:
                    with st.spinner("Generating predictions..."):
                        # Get sequences
                        X, _ = processor.create_sequences(sequence_length)
                        
                        # Properly scale sequences using model's scalers
                        if f'lr_model_X' in model_manager.scalers:
                            scaler_X = model_manager.scalers['lr_model_X']
                            X_flat = X.reshape(X.shape[0], -1)
                            X_scaled = scaler_X.transform(X_flat)
                            last_sequence = X_scaled[-1]
                        else:
                            # Fallback: reshape last sequence directly
                            last_sequence = X[-1].flatten()
                        
                        # Predict next values
                        predictions = model_manager.predict_next_values(
                            last_sequence,
                            num_steps=pred_steps
                        )
                        
                        st.success("✅ Predictions generated!")
                        
                        # Display predictions
                        pred_df = pd.DataFrame({
                            'Step': range(1, len(predictions) + 1),
                            'Predicted CPU (%)': predictions.round(2)
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Store for allocation
                        st.session_state.predictions = predictions
                else:
                    st.warning("Please load data and train model first")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("⚡ Allocate VMs Based on Predictions"):
            allocator = st.session_state.vm_allocator
            
            try:
                if hasattr(st.session_state, 'predictions'):
                    with st.spinner("Allocating VMs..."):
                        allocation_result = allocator.allocate_predictive(
                            st.session_state.predictions
                        )
                        
                        st.success("✅ Allocation completed!")
                        
                        col2_1, col2_2, col2_3 = st.columns(3)
                        with col2_1:
                            st.metric("Total VMs", allocation_result['total_vms'])
                        with col2_2:
                            st.metric("Successful", allocation_result['successful'])
                        with col2_3:
                            success_rate = (allocation_result['successful'] / 
                                          allocation_result['total_vms'] * 100)
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Store allocation result
                        st.session_state.allocation_result = allocation_result
                else:
                    st.warning("Please generate predictions first")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Visualization
    st.subheader("Prediction vs Allocation")
    
    try:
        if hasattr(st.session_state, 'predictions'):
            pred_df = pd.DataFrame({
                'Step': range(1, len(st.session_state.predictions) + 1),
                'Predicted CPU': st.session_state.predictions
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pred_df['Step'],
                y=pred_df['Predicted CPU'],
                mode='lines+markers',
                name='Predicted CPU',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="CPU Usage Predictions",
                xaxis_title="Prediction Step",
                yaxis_title="CPU Usage (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")

# ==================== TAB 4: METRICS & MONITORING ====================
with tab4:
    st.header("Real-time Metrics & Monitoring")
    
    tracker = st.session_state.metrics_tracker
    
    # Record sample metrics
    if st.button("📈 Record Sample Metrics"):
        try:
            # Simulate recording metrics
            for i in range(5):
                cpu = np.random.randint(30, 80)
                memory = np.random.randint(40, 75)
                storage = np.random.randint(25, 65)
                
                tracker.record_metric(
                    timestamp=i,
                    cpu_usage=cpu,
                    memory_usage=memory,
                    storage_usage=storage,
                    vms_count=i * 2,
                    active_hosts=5
                )
            
            st.success("✅ Metrics recorded!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    summary = tracker.get_summary()
    
    if summary:
        with col1:
            st.metric(
                "Performance Index",
                f"{tracker.get_performance_index():.1f}/100"
            )
        with col2:
            st.metric(
                "Health Status",
                tracker.get_health_status()
            )
        with col3:
            st.metric(
                "Avg CPU Usage",
                f"{summary['cpu']['mean']:.1f}%"
            )
        with col4:
            st.metric(
                "Total Records",
                summary['total_records']
            )
    
    # Recent metrics
    st.subheader("Recent Metrics")
    
    recent = tracker.get_recent_metrics(10)
    
    if not recent.empty:
        st.dataframe(recent, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(
                recent,
                x='timestamp',
                y=['cpu_usage', 'memory_usage', 'storage_usage'],
                title='Resource Utilization Over Time'
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                recent,
                x='timestamp',
                y='vms_count',
                title='VM Count Over Time'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Record metrics to see data")
    
    # Alerts
    st.subheader("System Alerts")
    
    alerts = tracker.get_alerts(5)
    
    if alerts:
        for alert in alerts:
            st.markdown(
                f"""<div class="alert-box">
                <strong>⚠️ {alert['datetime'].strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
                {alert['message']}
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No alerts")

# ==================== TAB 5: SYSTEM INFO ====================
with tab5:
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Datacenter Status")
        
        allocator = st.session_state.vm_allocator
        util = allocator.get_datacenter_utilization()
        
        metrics_display = {
            "Total Hosts": len(allocator.hosts),
            "Total VMs": util['total_vms'],
            "CPU Utilization": f"{util['cpu_utilization']:.2f}%",
            "Memory Utilization": f"{util['memory_utilization']:.2f}%",
            "Storage Utilization": f"{util['storage_utilization']:.2f}%",
            "Available Hosts": util['hosts_with_available_resources']
        }
        
        for metric, value in metrics_display.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("🤖 Model Information")
        
        model_manager = st.session_state.model_manager
        
        if "lr_model" in model_manager.models:
            info = model_manager.get_model_info("lr_model")
            st.json(info)
        else:
            st.info("No model trained yet")
    
    # Host details
    st.subheader("Host Details")
    
    host_stats = allocator.get_host_statistics()
    st.dataframe(host_stats, use_container_width=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=host_stats['host_id'],
        y=host_stats['cpu_utilization'],
        name='CPU'
    ))
    
    fig.add_trace(go.Bar(
        x=host_stats['host_id'],
        y=host_stats['memory_utilization'],
        name='Memory'
    ))
    
    fig.update_layout(
        title="Host Resource Utilization",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <p style='font-size: 12px;'>
        🖥️ <strong>VM Scheduler CloudSim</strong> | 
        Predictive VM Allocation using ML | 
        Built with Streamlit, Scikit-learn & SimPy
    </p>
</div>
""", unsafe_allow_html=True)
