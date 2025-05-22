import subprocess
import sys
import os

def main():
    """Install dependencies and prepare environment for Mimir"""
    print("Setting up Mimir environment...")
    
    # Install dependencies from requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install additional packages for TinyLlama compatibility
    extra_packages = [
        "einops>=0.7.0",
        "accelerate>=0.27.2", 
        "safetensors>=0.4.2"
    ]
    
    print("Installing extra packages for TinyLlama compatibility...")
    for package in extra_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Setup complete! You can now run Mimir with 'python run.py'")

if __name__ == "__main__":
    main() 