"""
Quick Start Script for VM Scheduler CloudSim GUI
Run this script to start the interactive dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("🖥️  VM Scheduler CloudSim - Interactive Dashboard Launcher")
    print("=" * 70)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        sys.exit(1)
    
    print("✅ Python version OK")
    print()
    
    # Check and install dependencies
    print("📦 Checking dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print(f"Installing requirements from {requirements_file}...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            sys.exit(1)
    else:
        print("⚠️  requirements.txt not found")
        print("Please ensure all required packages are installed:")
        print("  pip install streamlit pandas numpy scikit-learn plotly scipy simpy")
    
    print()
    print("=" * 70)
    print("🚀 Starting Streamlit Dashboard...")
    print("=" * 70)
    print()
    print("The dashboard will open in your default browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Get the path to gui_app.py
    gui_app = Path(__file__).parent / "gui_app.py"
    
    if not gui_app.exists():
        print(f"❌ Error: {gui_app} not found")
        sys.exit(1)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(gui_app),
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
