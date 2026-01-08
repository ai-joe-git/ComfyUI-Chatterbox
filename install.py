"""
Installation script for ComfyUI-Chatterbox
Handles CPU-safe installation
"""

import subprocess
import sys

def install():
    print("="*60)
    print("Installing ComfyUI-Chatterbox")
    print("="*60)
    print()
    
    print("1/2 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "-r", "requirements.txt"
    ])
    
    print("\n2/2 Installing chatterbox-tts (CPU-safe)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "chatterbox-tts", "--no-deps"
    ])
    
    print("\n" + "="*60)
    print("✅ Installation complete!")
    print("="*60)
    print("\n📝 Next steps:")
    print("   1. Login to HuggingFace:")
    print("      huggingface-cli login")
    print("   2. Restart ComfyUI")
    print()

if __name__ == "__main__":
    install()
