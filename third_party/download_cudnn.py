import os
import shutil
import subprocess
import re
import urllib.request
import ssl
import sys

def get_cuda_version():
    """Detect CUDA version from nvcc or environment."""
    try:
        # Try finding nvcc
        nvcc_path = shutil.which("nvcc")
        if not nvcc_path:
            # Common Windows path check
            default_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
            if os.path.exists(default_path):
                nvcc_path = default_path
        
        if nvcc_path:
            result = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            match = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
    except Exception as e:
        print(f"Error checking CUDA version: {e}")
    
    # Fallback: Check environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        match = re.search(r"v(\d+)\.(\d+)", cuda_path)
        if match:
            return int(match.group(1)), int(match.group(2))

    return None

def download_cudnn():
    cuda_ver = get_cuda_version()
    if cuda_ver:
        print(f"Detected CUDA version: {cuda_ver[0]}.{cuda_ver[1]}")
    else:
        print("Could not detect CUDA version. Assuming compatible version or manual download required.")
    
    # Note: NVIDIA requires login for most direct downloads, or specific signed URLs.
    # This is a placeholder logic that would be populated with valid URLs if available publicly.
    # For now, we will assume user might have local installation or we print instructions.
    
    # Check if already installed in standard location (Windows)
    local_path = r"C:\Program Files\NVIDIA\CUDNN\v9.17"
    if os.path.exists(local_path):
        print(f"Found local cuDNN installation at {local_path}. Skipping download.")
        return

    print("Note: Automated cuDNN download often requires NVIDIA Developer Program login.")
    print("Please install cuDNN manually if not found automatically.")
    
    # Logic for downloading if a public mirror existed:
    # url = "..."
    # ... download code similar to onednn ...

if __name__ == "__main__":
    download_cudnn()
