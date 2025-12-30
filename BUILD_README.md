# AudioVideoCaption - Executable Build

## 📦 Build Information

The AudioVideoCaption application has been successfully compiled into a standalone executable file.

**File Location:** `dist\AudioVideoCaption.exe`  
**File Size:** ~377 MB

## 🚀 How to Use

1. **Locate the executable:**
   - Navigate to the `dist` folder
   - Find `AudioVideoCaption.exe`

2. **Run the application:**
   - Double-click `AudioVideoCaption.exe` to launch
   - No Python installation required!

## ⚠️ Important Requirements

### FFmpeg Installation (REQUIRED)

The application requires FFmpeg to be installed and available in your system PATH:

1. **Download FFmpeg:**
   - Visit: https://ffmpeg.org/download.html
   - For Windows: https://www.gyan.dev/ffmpeg/builds/

2. **Install FFmpeg:**
   - Extract the downloaded archive
   - Add the `bin` folder to your system PATH
   - Verify installation by opening Command Prompt and typing: `ffmpeg -version`

### GPU Acceleration (OPTIONAL)

For best performance with video rendering and Whisper transcription:
- NVIDIA GPU with CUDA support is recommended
- Install appropriate CUDA drivers from NVIDIA

## 📋 Features

The executable includes all features of the AudioVideoCaption application:

- **Audio-Video Merging:** Merge MP3 audio files with MP4 video files
- **Caption Generation:** Generate captions using OpenAI Whisper
- **Caption Burning:** Burn captions directly into videos
- **Batch Processing:** Process multiple files at once
- **Customizable Captions:** Adjust font size and color

## 🔧 First Run Notes

- **First launch may be slower** as the application extracts temporary files
- **Whisper models** will be downloaded on first use (requires internet connection)
- **Antivirus software** may flag the executable - this is a false positive common with PyInstaller executables

## 📁 Distribution

To distribute the application:

1. **Single File Distribution:**
   - Simply copy `AudioVideoCaption.exe` to the target computer
   - Ensure FFmpeg is installed on the target system

2. **Portable Package:**
   - Create a folder containing:
     - `AudioVideoCaption.exe`
     - FFmpeg binaries (optional, for portability)
     - README with instructions

## 🐛 Troubleshooting

### Application won't start
- Check if antivirus is blocking the executable
- Run as Administrator if needed
- Ensure you have sufficient disk space (~1GB free)

### FFmpeg errors
- Verify FFmpeg is installed: `ffmpeg -version` in Command Prompt
- Ensure FFmpeg is in system PATH
- Try reinstalling FFmpeg

### Whisper model download fails
- Check internet connection
- Ensure firewall allows the application to access the internet
- Models are downloaded to: `C:\Users\<YourName>\.cache\whisper`

### Performance issues
- Close other applications to free up RAM
- For large files, ensure sufficient disk space
- Consider using smaller Whisper models (tiny/base instead of medium)

## 📝 Rebuilding the Executable

If you need to rebuild the executable:

```bash
cd d:\CodeTool\AudioVideoCaption
python build_exe.py
```

The build process will:
1. Install PyInstaller (if not already installed)
2. Bundle all dependencies
3. Create the executable in the `dist` folder

## 📄 License & Credits

- Application: AudioVideoCaption
- Built with: PyInstaller, PyQt6, OpenAI Whisper, FFmpeg
- Build Date: 2025-12-30

---

**Enjoy using AudioVideoCaption! 🎬**
