"""
Build script for AudioVideoCaption
Creates a standalone .exe file using PyInstaller
"""

import os
import sys
import subprocess
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    print("📦 Checking PyInstaller...")
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📥 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed successfully")

def build_exe():
    """Build the executable using PyInstaller"""
    print("\n🔨 Building AudioVideoCaption.exe...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=AudioVideoCaption",
        "--onefile",                    # Single exe file
        "--windowed",                   # No console window
        "--clean",                      # Clean cache
        "--add-data=settings.json;.",   # Include settings file
        "--add-data=queue_data.json;.", # Include queue data file
        "--hidden-import=whisper",      # Ensure whisper is included
        "--hidden-import=imageio",      # Ensure imageio is included
        "--hidden-import=PyQt6",        # Ensure PyQt6 is included
        "--collect-all=whisper",        # Collect all whisper files
        "--collect-all=imageio",        # Collect all imageio files
        "ui.py"                         # Main entry point
    ]
    
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, cwd=script_dir)
        print("\n✅ Build completed successfully!")
        print(f"📁 Executable location: {os.path.join(script_dir, 'dist', 'AudioVideoCaption.exe')}")
        print("\n🎉 You can now run AudioVideoCaption.exe from the dist folder!")
        print("\n⚠️  IMPORTANT NOTES:")
        print("   - FFmpeg must be installed and in PATH for the app to work!")
        print("   - The first run may take longer as Whisper downloads models")
        print("   - Ensure you have a CUDA-capable GPU for best performance")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("  AudioVideoCaption - Build to EXE")
    print("=" * 60)
    
    # Step 1: Install PyInstaller
    install_pyinstaller()
    
    # Step 2: Build executable
    build_exe()
    
    print("\n" + "=" * 60)
    print("  Build process completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
