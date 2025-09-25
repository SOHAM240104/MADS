#!/usr/bin/env python3
"""
GPU Configuration Optimizer for Seq2Seq Model
Automatically detects hardware and optimizes config.py settings
"""

import torch
import os
import sys
import subprocess

def detect_system():
    """Detect system capabilities and recommend configuration"""
    
    print("🔍 Detecting system capabilities...")
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Get system info
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_type = "CUDA"
    elif mps_available:
        gpu_type = "MPS"
        gpu_memory = 8  # Approximate for Apple Silicon
        gpu_name = "Apple Silicon"
    else:
        gpu_type = "CPU"
        gpu_memory = 0
        gpu_name = "CPU Only"
    
    print(f"🖥️  Device: {gpu_name}")
    print(f"🚀 Acceleration: {gpu_type}")
    if gpu_memory > 0:
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    return gpu_type, gpu_memory, gpu_name

def recommend_config(gpu_type, gpu_memory):
    """Recommend optimal configuration based on hardware"""
    
    print("\n⚙️  Recommended configuration:")
    
    if gpu_type == "CUDA":
        if gpu_memory >= 24:  # RTX 4090, A100, etc.
            config = {
                "BATCH_SIZE": 128,
                "LEARNING_RATE": 0.005,
                "LSTM_HIDDEN_DIMENSION": 768,
                "LSTM_LAYERS": 3,
                "N_EPOCHS": 15
            }
            time_estimate = "15-25 minutes"
        elif gpu_memory >= 12:  # RTX 3080 Ti, 4070 Ti, etc.
            config = {
                "BATCH_SIZE": 64,
                "LEARNING_RATE": 0.003,
                "LSTM_HIDDEN_DIMENSION": 512,
                "LSTM_LAYERS": 2,
                "N_EPOCHS": 12
            }
            time_estimate = "25-40 minutes"
        elif gpu_memory >= 8:  # RTX 3070, 4060 Ti, etc.
            config = {
                "BATCH_SIZE": 32,
                "LEARNING_RATE": 0.002,
                "LSTM_HIDDEN_DIMENSION": 512,
                "LSTM_LAYERS": 2,
                "N_EPOCHS": 10
            }
            time_estimate = "35-50 minutes"
        else:  # Lower-end GPUs
            config = {
                "BATCH_SIZE": 16,
                "LEARNING_RATE": 0.001,
                "LSTM_HIDDEN_DIMENSION": 256,
                "LSTM_LAYERS": 2,
                "N_EPOCHS": 10
            }
            time_estimate = "60-90 minutes"
    
    elif gpu_type == "MPS":  # Apple Silicon
        config = {
            "BATCH_SIZE": 16,
            "LEARNING_RATE": 0.001,
            "LSTM_HIDDEN_DIMENSION": 512,
            "LSTM_LAYERS": 2,
            "N_EPOCHS": 10
        }
        time_estimate = "90-120 minutes"
    
    else:  # CPU only
        config = {
            "BATCH_SIZE": 4,
            "LEARNING_RATE": 0.001,
            "LSTM_HIDDEN_DIMENSION": 256,
            "LSTM_LAYERS": 1,
            "N_EPOCHS": 8
        }
        time_estimate = "4-6 hours"
    
    print(f"📊 Batch Size: {config['BATCH_SIZE']}")
    print(f"🎯 Learning Rate: {config['LEARNING_RATE']}")
    print(f"🧠 Hidden Dimension: {config['LSTM_HIDDEN_DIMENSION']}")
    print(f"📚 LSTM Layers: {config['LSTM_LAYERS']}")
    print(f"🔄 Epochs: {config['N_EPOCHS']}")
    print(f"⏱️  Estimated Time: {time_estimate}")
    
    return config

def update_config_file(config):
    """Update config.py with recommended settings"""
    
    print("\n📝 Updating config.py...")
    
    config_path = "config.py"
    if not os.path.exists(config_path):
        print("❌ config.py not found!")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update relevant lines
    updated_lines = []
    for line in lines:
        if line.strip().startswith('BATCH_SIZE ='):
            updated_lines.append(f"BATCH_SIZE = {config['BATCH_SIZE']}\n")
        elif line.strip().startswith('LEARNING_RATE='):
            updated_lines.append(f"LEARNING_RATE={config['LEARNING_RATE']}\n")
        elif line.strip().startswith('LSTM_HIDDEN_DIMENSION ='):
            updated_lines.append(f"LSTM_HIDDEN_DIMENSION = {config['LSTM_HIDDEN_DIMENSION']}\n")
        elif line.strip().startswith('LSTM_LAYERS='):
            updated_lines.append(f"LSTM_LAYERS={config['LSTM_LAYERS']}\n")
        elif line.strip().startswith('N_EPOCHS ='):
            updated_lines.append(f"N_EPOCHS = {config['N_EPOCHS']}\n")
        else:
            updated_lines.append(line)
    
    # Write updated config
    with open(f"{config_path}.backup", 'w') as f:
        f.writelines(lines)  # Backup original
    
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("✅ config.py updated (backup saved as config.py.backup)")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    
    print("\n🔍 Checking dependencies...")
    
    required_packages = ['torch', 'numpy', 'scikit-learn', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available!")
    return True

def main():
    print("🚀 Seq2Seq Model GPU Configuration Optimizer")
    print("=" * 50)
    
    # Detect system
    gpu_type, gpu_memory, gpu_name = detect_system()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get recommendations
    config = recommend_config(gpu_type, gpu_memory)
    
    # Ask user if they want to update config
    print(f"\n❓ Update config.py with these settings? (y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        if update_config_file(config):
            print("\n🎉 Configuration optimized for your system!")
            print("You can now run: python train_phased.py")
        else:
            print("❌ Failed to update config.py")
    else:
        print("\n📝 Manual configuration recommended:")
        print("Update these values in config.py:")
        for key, value in config.items():
            print(f"  {key} = {value}")
    
    print("\n💡 Tips:")
    print("- Monitor GPU usage during training")
    print("- Reduce batch size if you get out-of-memory errors") 
    print("- Increase batch size if GPU utilization is low")
    print("- Check logs/ directory for training progress")

if __name__ == "__main__":
    main()