import os
import sys
import subprocess
import tensorflow as tf
import torch  # We'll try to use torch for CUDA detection as well
import platform

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed"""
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        print("\n=== NVIDIA Driver Info ===")
        print(output.decode())
        return True
    except:
        print("\n=== NVIDIA Driver Status ===")
        print("NVIDIA drivers not found or not properly installed")
        print("Please download and install from: https://www.nvidia.com/Download/index.aspx")
        return False

def check_cuda():
    """Check CUDA installation"""
    print("\n=== CUDA Status ===")
    
    # Check CUDA environment variables
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"CUDA_PATH is set to: {cuda_path}")
    else:
        print("CUDA_PATH environment variable not set")
    
    # Check if CUDA is available through PyTorch (more reliable detection)
    print("\nCUDA Detection through PyTorch:")
    print(f"CUDA available: {'YES' if torch.cuda.is_available() else 'NO'}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

def check_cudnn():
    """Check cuDNN status"""
    print("\n=== cuDNN Status ===")
    try:
        import ctypes
        cudnn = ctypes.CDLL('cudnn64_8.dll')
        print("cuDNN is installed")
    except:
        print("cuDNN not found or not properly installed")
        print("Download cuDNN from: https://developer.nvidia.com/cudnn")

def check_tensorflow():
    """Check TensorFlow GPU support"""
    print("\n=== TensorFlow GPU Status ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    physical_devices = tf.config.list_physical_devices()
    print("\nAvailable TensorFlow Devices:")
    for device in physical_devices:
        print(f"- {device.device_type}: {device.name}")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("\nTensorFlow GPU support is enabled")
        for gpu in gpu_devices:
            print(f"  Found GPU: {gpu}")
    else:
        print("\nNo GPU devices found by TensorFlow")

def check_system_info():
    """Check system information"""
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.version()}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")

def main():
    print("=== GPU Setup Diagnostic Tool ===")
    
    check_system_info()
    driver_status = check_nvidia_driver()
    check_cuda()
    check_cudnn()
    check_tensorflow()
    
    print("\n=== Summary of Required Actions ===")
    if not driver_status:
        print("1. Install NVIDIA drivers")
    if 'CUDA_PATH' not in os.environ:
        print("2. Install CUDA Toolkit 11.8")
    if not tf.config.list_physical_devices('GPU'):
        print("3. Verify CUDA and cuDNN installation")
        print("4. Ensure TensorFlow is properly installed with GPU support")

if __name__ == "__main__":
    main() 