"""CUDA diagnostic script to help troubleshoot GPU detection issues."""

import sys
import subprocess

print("="*70)
print("CUDA DIAGNOSTIC TOOL")
print("="*70)

# 1. Check NVIDIA GPU
print("\n1. Checking for NVIDIA GPU...")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   [OK] nvidia-smi found:")
        print("\n" + result.stdout)
    else:
        print("   [FAIL] nvidia-smi returned error")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/download/index.aspx")
except FileNotFoundError:
    print("   [FAIL] nvidia-smi not found")
    print("   This means NVIDIA drivers are not installed or not in PATH")
    print("   Install from: https://www.nvidia.com/download/index.aspx")
except Exception as e:
    print(f"   [ERROR] {e}")

# 2. Check PyTorch installation
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   [OK] PyTorch version: {torch.__version__}")
    print(f"   CUDA compiled version: {torch.version.cuda}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Compute capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("\n   [ISSUE] CUDA not available in PyTorch")
        if torch.version.cuda is None:
            print("   Your PyTorch is CPU-only version")
            print("   Reinstall PyTorch with CUDA support:")
            print("   pip uninstall torch")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("   PyTorch has CUDA support but cannot find GPU")
            print("   Possible causes:")
            print("   - NVIDIA drivers not installed")
            print("   - GPU disabled in BIOS")
            print("   - Incompatible CUDA version")
except ImportError:
    print("   [FAIL] PyTorch not installed")
    print("   Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
except Exception as e:
    print(f"   [ERROR] {e}")

# 3. Check CUDA Toolkit
print("\n3. Checking CUDA Toolkit...")
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   [OK] CUDA Toolkit found:")
        print("   " + result.stdout.split('\n')[-2])
    else:
        print("   [WARN] nvcc not found (CUDA Toolkit not required for inference)")
except FileNotFoundError:
    print("   [WARN] nvcc not found (CUDA Toolkit not required for inference)")
except Exception as e:
    print(f"   [INFO] {e}")

# 4. Check cuDNN
print("\n4. Checking cuDNN...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("   [SKIP] Cannot check cuDNN without CUDA")
except Exception as e:
    print(f"   [INFO] {e}")

# 5. Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

try:
    import torch
    
    if not torch.cuda.is_available():
        print("\nYour system does not have CUDA available. To fix this:")
        print("\n1. Verify NVIDIA GPU:")
        print("   - Run 'nvidia-smi' in command prompt")
        print("   - If it fails, install NVIDIA drivers")
        print("\n2. Install PyTorch with CUDA:")
        print("   pip uninstall torch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("\n3. Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        print("\nIf you don't have an NVIDIA GPU, use CPU mode in the application.")
    else:
        print("\n[SUCCESS] Your CUDA setup is working correctly!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("You can use CUDA mode in the application.")
        
except ImportError:
    print("\nPyTorch is not installed. Install it with:")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*70)
