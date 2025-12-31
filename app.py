import os
import random
import subprocess
import math
import re
import sys
import time
import shlex
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import imageio.v3 as iio

# -------------------------
# Helpers
# -------------------------

def run(cmd: str):
    print(">>", cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        error_msg = f"Command failed with code {p.returncode}"
        if p.stderr:
            error_msg += f"\n\nFFmpeg Error Output:\n{p.stderr}"
        if p.stdout:
            error_msg += f"\n\nFFmpeg Standard Output:\n{p.stdout}"
        print(f"\n❌ {error_msg}")
        raise RuntimeError(error_msg)

def run_with_progress(cmd: str, total_duration: float):
    print(f">> {cmd}")
    
    # Use Popen to read output in real-time
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Regex to extract time and speed
    # time=00:00:08.53
    time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
    speed_pattern = re.compile(r"speed=\s*(\d+\.?\d*)x")
    
    print("\nRendering Progress:")
    print("-" * 60)
    
    all_output = []  # Collect all output for error reporting
    
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            all_output.append(line)  # Store for error reporting
                
            # Check for time
            t_match = time_pattern.search(line)
            s_match = speed_pattern.search(line)
            
            if t_match:
                h, m, s = map(float, t_match.groups())
                current_seconds = h * 3600 + m * 60 + s
                
                percent = min(100, (current_seconds / total_duration) * 100)
                
                speed = "N/A"
                eta = "N/A"
                
                if s_match:
                    speed_val = float(s_match.group(1))
                    speed = f"{speed_val:.2f}x"
                    if speed_val > 0:
                        remaining = (total_duration - current_seconds) / speed_val
                        eta = f"{int(remaining)}s"
                
                # Progress Bar
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Speed: {speed} | ETA: {eta}   ")
                sys.stdout.flush()
            
            # Also print errors if any (lines not matching stats often contain warnings/errors)
            elif "Error" in line or "failed" in line or "Invalid" in line:
                print(f"\n⚠️  {line}")

    except Exception as e:
        print(f"\nError reading progress: {e}")
        
    process.wait()
    print("\n" + "-" * 60)
    
    if process.returncode != 0:
        error_msg = f"FFmpeg failed with code {process.returncode}"
        # Show last 20 lines of output for debugging
        if all_output:
            error_msg += "\n\nLast FFmpeg output lines:\n" + "\n".join(all_output[-20:])
        print(f"\n❌ {error_msg}")
        raise RuntimeError(error_msg)

def get_audio_duration(audio_path: str) -> float:
    cmd = (
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{audio_path}"'
    )
    out = subprocess.check_output(cmd, shell=True).decode().strip()
    return float(out)

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = (
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    )
    out = subprocess.check_output(cmd, shell=True).decode().strip()
    return float(out)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    imgs = []
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            imgs.append(p.as_posix())
    if not imgs:
        raise ValueError("No images found in images_dir.")
    return imgs

def list_overlays(overlay_dir: str) -> List[str]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    files = []
    if not os.path.exists(overlay_dir):
        return []
    for p in sorted(Path(overlay_dir).iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p.as_posix())
    return files

# -------------------------
# Caption Generation (Whisper)
# -------------------------

def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_captions_whisper(
    audio_or_video_path: str,
    output_srt_path: str,
    model_name: str = "base",
    language: str = "en"
) -> str:
    """
    Generate SRT captions using OpenAI Whisper.
    
    Args:
        audio_or_video_path: Path to audio (MP3) or video (MP4) file
        output_srt_path: Path to save SRT file
        model_name: Whisper model (tiny, base, small, medium)
        language: Language code (en, vi, etc.) or None for auto-detect
    
    Returns:
        Path to generated SRT file
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed. Install with: pip install openai-whisper"
        )
    
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing audio from: {audio_or_video_path}")
    
    # Whisper can handle both audio and video files
    result = model.transcribe(
        audio_or_video_path,
        language=language if language != "auto" else None,
        verbose=True
    )
    
    # Generate SRT file
    print(f"Generating SRT file: {output_srt_path}")
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            start_time = format_timestamp_srt(segment['start'])
            end_time = format_timestamp_srt(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    print(f"✓ Captions saved to: {output_srt_path}")
    return output_srt_path

def burn_captions_to_video(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 2
) -> str:
    """
    Burn SRT captions into video using FFmpeg.
    
    Args:
        video_path: Input video file
        srt_path: SRT subtitle file
        output_path: Output video with burned captions
        font_size: Caption font size
        font_color: Caption text color
        outline_color: Caption outline color
        outline_width: Caption outline width
    
    Returns:
        Path to output video
    """
    # Convert paths to absolute and use forward slashes for FFmpeg
    srt_abs = Path(srt_path).resolve().as_posix()
    
    # Escape special characters in path for FFmpeg filter
    srt_escaped = srt_abs.replace('\\', '/').replace(':', '\\:')
    
    # Build subtitle filter with styling
    subtitle_filter = (
        f"subtitles='{srt_escaped}'"
        f":force_style='FontSize={font_size},"
        f"PrimaryColour=&H{color_to_ass_hex(font_color)},"
        f"OutlineColour=&H{color_to_ass_hex(outline_color)},"
        f"Outline={outline_width},"
        f"Alignment=2'"  # Bottom center
    )
    
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel warning '
        f'-i "{video_path}" '
        f'-vf "{subtitle_filter}" '
        f'-c:v h264_nvenc -preset p4 -b:v 2000k '
        f'-c:a copy '
        f'"{output_path}"'
    )
    
    print(f"Burning captions into video...")
    print(f">> {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to burn captions: FFmpeg error")
    
    print(f"✓ Video with captions saved to: {output_path}")
    return output_path

def color_to_ass_hex(color_name: str) -> str:
    """Convert color name to ASS subtitle format hex (AABBGGRR)"""
    colors = {
        'white': 'FFFFFF',
        'black': '000000',
        'yellow': '00FFFF',
        'red': '0000FF',
        'green': '00FF00',
        'blue': 'FF0000',
    }
    return colors.get(color_name.lower(), 'FFFFFF')


# -------------------------
# Audio-Video Merging
# -------------------------

def merge_audio_video_files(
    mp3_path: str,
    mp4_path: str,
    output_path: str,
    generate_captions: bool = False,
    whisper_model: str = "tiny",
    burn_captions: bool = False,
    caption_color: str = "white",
    caption_font_size: int = 24
) -> str:
    """
    Merge MP3 audio with MP4 video.
    
    Args:
        mp3_path: Path to MP3 audio file
        mp4_path: Path to MP4 video file
        output_path: Path for output video
        generate_captions: Whether to generate captions from audio
        whisper_model: Whisper model to use (tiny, base, small, medium)
        burn_captions: Whether to burn captions into video
    
    Returns:
        Path to output video
    """
    print(f"\n{'='*60}")
    print(f"Merging: {Path(mp3_path).name} + {Path(mp4_path).name}")
    print(f"Output: {Path(output_path).name}")
    print(f"{'='*60}\n")
    
    # Step 1: Merge audio and video
    temp_merged = str(Path(output_path).with_suffix('.temp.mp4'))
    
    cmd_merge = (
        f'ffmpeg -y -hide_banner -loglevel warning '
        f'-i "{mp4_path}" -i "{mp3_path}" '
        f'-c:v copy -c:a aac -b:a 192k '
        f'-map 0:v:0 -map 1:a:0 '
        f'-shortest '
        f'"{temp_merged}"'
    )
    
    print(">> Merging audio and video...")
    run(cmd_merge)
    print("✓ Audio and video merged\n")
    
    # Step 2: Generate captions if requested
    if generate_captions:
        srt_path = str(Path(output_path).with_suffix('.srt'))
        print(f">> Generating captions using Whisper ({whisper_model})...")
        generate_captions_whisper(
            mp3_path,  # Use MP3 for faster processing
            srt_path,
            whisper_model,
            language="auto"
        )
        
        # Step 3: Burn captions if requested
        if burn_captions:
            print(">> Burning captions into video...")
            burn_captions_to_video(
                temp_merged,
                srt_path,
                output_path,
                font_size=caption_font_size,
                font_color=caption_color
            )
            # Clean up temp merged file and SRT file
            try:
                os.remove(temp_merged)
                os.remove(srt_path)  # Delete SRT file
            except OSError:
                pass
        else:
            # No burning, just rename temp to final
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_merged, output_path)
            except OSError as e:
                print(f"Warning: Could not rename temp file: {e}")
    else:
        # No captions, just rename temp to final
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_merged, output_path)
        except OSError as e:
            print(f"Warning: Could not rename temp file: {e}")
    
    print(f"\n✓ Completed: {Path(output_path).name}\n")
    return output_path



def process_audio_video_batch(
    audio_folder: str,
    video_folder: str,
    output_folder: str,
    generate_captions: bool = False,
    whisper_model: str = "tiny",
    burn_captions: bool = False,
    caption_color: str = "white",
    caption_font_size: int = 24
) -> List[Tuple[str, bool, str]]:
    """
    Process all matched MP3-MP4 pairs in folders.
    
    Args:
        audio_folder: Folder containing MP3 files
        video_folder: Folder containing MP4 files
        output_folder: Folder for output videos
        generate_captions: Whether to generate captions
        whisper_model: Whisper model to use
        burn_captions: Whether to burn captions
    
    Returns:
        List of (filename, success, message) tuples
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Scan for MP3 files
    audio_path = Path(audio_folder)
    mp3_files = {f.stem: f for f in audio_path.glob("*.mp3")}
    mp3_files.update({f.stem: f for f in audio_path.glob("*.MP3")})
    
    # Scan for MP4 files
    video_path = Path(video_folder)
    mp4_files = {f.stem: f for f in video_path.glob("*.mp4")}
    mp4_files.update({f.stem: f for f in video_path.glob("*.MP4")})
    
    print(f"\nFound {len(mp3_files)} MP3 file(s)")
    print(f"Found {len(mp4_files)} MP4 file(s)\n")
    
    # Match files by name
    matched_pairs = []
    for name in mp3_files.keys():
        if name in mp4_files:
            matched_pairs.append((name, mp3_files[name], mp4_files[name]))
    
    print(f"Matched {len(matched_pairs)} file pair(s)\n")
    
    if not matched_pairs:
        print("⚠️  No matched file pairs found!")
        return []
    
    # Warn about unmatched files
    unmatched_mp3 = set(mp3_files.keys()) - set(mp4_files.keys())
    unmatched_mp4 = set(mp4_files.keys()) - set(mp3_files.keys())
    
    if unmatched_mp3:
        print(f"⚠️  Unmatched MP3 files (no matching MP4): {', '.join(unmatched_mp3)}")
    if unmatched_mp4:
        print(f"⚠️  Unmatched MP4 files (no matching MP3): {', '.join(unmatched_mp4)}")
    if unmatched_mp3 or unmatched_mp4:
        print()
    
    # Process each pair
    results = []
    for idx, (name, mp3_file, mp4_file) in enumerate(matched_pairs, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(matched_pairs)}: {name}")
        print(f"{'='*60}")
        
        output_file = Path(output_folder) / f"{name}_output.mp4"
        
        try:
            merge_audio_video_files(
                str(mp3_file),
                str(mp4_file),
                str(output_file),
                generate_captions,
                whisper_model,
                burn_captions,
                caption_color,
                caption_font_size
            )
            results.append((name, True, "Success"))
            print(f"✓ [{idx}/{len(matched_pairs)}] {name} - Success")
        except Exception as e:
            error_msg = str(e)
            results.append((name, False, error_msg))
            print(f"✗ [{idx}/{len(matched_pairs)}] {name} - Error: {error_msg}")
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete!")
    print(f"Successful: {sum(1 for _, success, _ in results if success)}/{len(results)}")
    print(f"{'='*60}\n")
    
    return results


def process_burn_captions_batch(
    video_folder: str,
    srt_folder: str,
    output_folder: str,
    font_size: int = 24,
    font_color: str = "white"
) -> List[Tuple[str, bool, str]]:
    """
    Process batch burning of captions into videos.
    
    Args:
        video_folder: Folder containing MP4 videos
        srt_folder: Folder containing SRT subtitles
        output_folder: Folder for output videos
        font_size: Caption font size
        font_color: Caption color
        
    Returns:
        List of (filename, success, message) tuples
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Scan for MP4 files
    video_path = Path(video_folder)
    mp4_files = {f.stem: f for f in video_path.glob("*.mp4")}
    mp4_files.update({f.stem: f for f in video_path.glob("*.MP4")})
    
    # Scan for SRT files
    srt_path = Path(srt_folder)
    srt_files = {f.stem: f for f in srt_path.glob("*.srt")}
    srt_files.update({f.stem: f for f in srt_path.glob("*.SRT")})
    
    print(f"\nFound {len(mp4_files)} MP4 file(s)")
    print(f"Found {len(srt_files)} SRT file(s)\n")
    
    # Match files by name
    matched_pairs = []
    for name in mp4_files.keys():
        if name in srt_files:
            matched_pairs.append((name, mp4_files[name], srt_files[name]))
            
    print(f"Matched {len(matched_pairs)} file pair(s)\n")
    
    if not matched_pairs:
        print("⚠️  No matched file pairs found!")
        return []

    # Process each pair
    results = []
    for idx, (name, mp4_file, srt_file) in enumerate(matched_pairs, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(matched_pairs)}: {name}")
        print(f"{'='*60}")
        
        output_file = Path(output_folder) / f"{name}_captioned.mp4"
        
        try:
            burn_captions_to_video(
                str(mp4_file),
                str(srt_file),
                str(output_file),
                font_size=font_size,
                font_color=font_color
            )
            results.append((name, True, "Success"))
            print(f"✓ [{idx}/{len(matched_pairs)}] {name} - Success")
        except Exception as e:
            error_msg = str(e)
            results.append((name, False, error_msg))
            print(f"✗ [{idx}/{len(matched_pairs)}] {name} - Error: {error_msg}")
            
    print(f"\n{'='*60}")
    print(f"Batch Burn Processing Complete!")
    print(f"Successful: {sum(1 for _, success, _ in results if success)}/{len(results)}")
    print(f"{'='*60}\n")
    
    return results



# -------------------------
# Clip Storage (Filesystem-Based)
# -------------------------

def ensure_storage_dir(storage_dir: str):
    """Ensure storage directory exists"""
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

def parse_clip_filename(filename: str) -> Optional[Tuple[int, float]]:
    """
    Parse clip filename to extract ID and duration.
    Format: clip_0001_140.7s.mp4
    Returns: (clip_id, duration) or None if invalid
    """
    import re
    match = re.match(r'clip_(\d+)_([\d.]+)s\.mp4$', filename)
    if match:
        clip_id = int(match.group(1))
        duration = float(match.group(2))
        return (clip_id, duration)
    return None

def scan_storage_clips(storage_dir: str) -> List[Tuple[str, float]]:
    """
    Scan storage directory for clips.
    Returns: List of (filepath, duration) tuples, sorted by clip ID
    """
    if not os.path.exists(storage_dir):
        return []
    
    clips = []
    for filepath in Path(storage_dir).glob("clip_*.mp4"):
        parsed = parse_clip_filename(filepath.name)
        if parsed:
            clip_id, duration = parsed
            clips.append((str(filepath), duration, clip_id))
    
    # Sort by clip ID (FIFO order)
    clips.sort(key=lambda x: x[2])
    
    # Return only filepath and duration
    return [(path, dur) for path, dur, _ in clips]

def generate_clip_storage(
    count: int,
    storage_dir: str,
    images_dir: str,
    overlays_dir: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    seg_min: int = 90,
    seg_max: int = 150,
    seed: int = 0,
    max_workers: int = 4  # Number of parallel threads
) -> List[str]:
    """
    Pre-generate video clips and store them for later use.
    Uses multi-threading for parallel generation.
    Filename format: clip_0001_140.7s.mp4 (includes duration)
    
    Args:
        count: Number of clips to generate
        storage_dir: Directory to store clips
        images_dir: Directory containing images
        overlays_dir: Directory containing overlay videos
        width, height: Video dimensions
        fps: Frames per second
        seg_min, seg_max: Min/max duration for each clip (seconds)
        seed: Random seed for reproducibility
        max_workers: Number of parallel threads (default: 4)
    
    Returns:
        List of generated clip file paths
    """
    print(f"Generating {count} clips for storage using {max_workers} threads...")
    ensure_storage_dir(storage_dir)
    
    imgs = list_images(images_dir)
    overlays = list_overlays(overlays_dir)
    
    # Scan existing clips to get next ID
    existing_clips = scan_storage_clips(storage_dir)
    next_id = 1
    if existing_clips:
        # Extract max ID from existing clips
        max_id = 0
        for filepath, _ in existing_clips:
            parsed = parse_clip_filename(Path(filepath).name)
            if parsed:
                clip_id, _ = parsed
                max_id = max(max_id, clip_id)
        next_id = max_id + 1
    
    # Prepare clip generation tasks
    tasks = []
    for i in range(count):
        clip_id = next_id + i
        
        # Use timestamp + clip ID for true randomization
        timestamp_seed = int(time.time() * 1000) + clip_id + random.randint(0, 100000)
        clip_rnd = random.Random(timestamp_seed)
        
        # Random duration
        duration = clip_rnd.uniform(seg_min, seg_max)
        
        # Filename with embedded duration
        clip_filename = f"clip_{clip_id:04d}_{duration:.1f}s.mp4"
        clip_path = str(Path(storage_dir) / clip_filename)
        
        # Random image and overlay
        image = clip_rnd.choice(imgs)
        overlay = clip_rnd.choice(overlays) if overlays else None
        
        tasks.append({
            'index': i,
            'clip_id': clip_id,
            'filename': clip_filename,
            'path': clip_path,
            'image': image,
            'overlay': overlay,
            'duration': duration,
            'seed': timestamp_seed,
            'width': width,
            'height': height,
            'fps': fps
        })
    
    # Generate clips in parallel
    generated_clips = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_render_single_clip_task, task): task
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                clip_path = future.result()
                generated_clips.append(clip_path)
                completed += 1
                print(f"  [{completed}/{count}] ✓ {task['filename']}")
            except Exception as e:
                print(f"  [{completed+1}/{count}] ✗ {task['filename']} - Error: {e}")
    
    print(f"✓ Generated {len(generated_clips)} clips in {storage_dir}")
    
    return generated_clips

def _render_single_clip_task(task: Dict) -> str:
    """Helper function to render a single clip (for threading)"""
    _render_single_clip(
        image=task['image'],
        overlay=task['overlay'],
        duration=task['duration'],
        out_path=task['path'],
        width=task['width'],
        height=task['height'],
        fps=task['fps'],
        seed=task['seed']
    )
    return task['path']

def _render_single_clip(
    image: str,
    overlay: Optional[str],
    duration: float,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    seed: int
):
    """Render a single clip (one image with Ken Burns + optional overlay)"""
    # Build filtergraph for single clip
    playlist = [image]
    seg_plan = [duration]
    overlays = [overlay] if overlay else []
    
    filter_complex, input_args = build_filtergraph(
        playlist, seg_plan, seed, width, height, fps, crossfade=0, overlays=overlays
    )
    
    # Use unique temp filenames based on output path (thread-safe)
    base_name = Path(out_path).stem
    filter_script_path = f"temp_filter_{base_name}.txt"
    batch_file_path = f"temp_render_{base_name}.bat"
    
    # Write filter script
    with open(filter_script_path, "w", encoding="utf-8") as f:
        f.write(filter_complex)
    
    inputs_str = " ".join(input_args)
    
    # Build FFmpeg command
    cmd_parts = ["ffmpeg"]
    cmd_parts.extend(["-y", "-hide_banner", "-loglevel", "error"])
    cmd_parts.extend(shlex.split(inputs_str))
    cmd_parts.extend(["-filter_complex_script", filter_script_path])
    cmd_parts.extend(["-map", "[video]"])
    cmd_parts.extend(["-r", str(fps)])
    cmd_parts.extend(["-t", f"{duration:.3f}"])
    cmd_parts.extend(["-c:v", "h264_nvenc"])
    cmd_parts.extend(["-preset", "p4"])
    cmd_parts.extend(["-tune", "hq"])
    cmd_parts.extend(["-b:v", "1500k"])
    cmd_parts.extend(["-maxrate", "1500k"])
    cmd_parts.extend(["-bufsize", "3000k"])
    cmd_parts.extend(["-pix_fmt", "yuv420p"])
    cmd_parts.append(f'"{out_path}"')
    
    # Write batch file
    with open(batch_file_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        cmd_line = " ".join(
            f'"{arg}"' if (" " in arg and not arg.startswith('"')) else arg
            for arg in cmd_parts
        )
        f.write(cmd_line + "\n")
    
    # Execute
    result = subprocess.run(batch_file_path, shell=True, capture_output=True)
    
    # Cleanup
    try:
        os.remove(filter_script_path)
        os.remove(batch_file_path)
    except OSError:
        pass
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate clip: {out_path}")

def get_clips_from_storage(
    required_duration: float,
    storage_dir: str
) -> Tuple[List[Tuple[str, float]], float]:
    """
    Get clips from storage to fill the required duration.
    Uses filesystem scanning - no metadata needed!
    
    Args:
        required_duration: Total duration needed (seconds)
        storage_dir: Storage directory path
    
    Returns:
        Tuple of (list of (filepath, duration) tuples, remaining duration needed)
    """
    # Scan storage directory
    all_clips = scan_storage_clips(storage_dir)
    
    if not all_clips:
        return [], required_duration
    
    # Select clips to fill duration (FIFO order - already sorted by ID)
    selected_clips = []
    total_duration = 0.0
    
    for filepath, duration in all_clips:
        if total_duration >= required_duration:
            break
        selected_clips.append((filepath, duration))
        total_duration += duration
    
    remaining = max(0, required_duration - total_duration)
    
    if selected_clips:
        print(f"Found {len(selected_clips)} clips in storage ({total_duration:.1f}s)")
    if remaining > 0:
        print(f"  Need {remaining:.1f}s more clips")
    
    return selected_clips, remaining

def delete_used_clips(clip_paths: List[str], storage_dir: str):
    """
    Delete used clips from storage.
    No metadata needed - just delete the files!
    
    Args:
        clip_paths: List of clip file paths to delete
        storage_dir: Storage directory path (unused, kept for compatibility)
    """
    if not clip_paths:
        return
    
    deleted_count = 0
    for clip_path in clip_paths:
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not delete {clip_path}: {e}")
    
    print(f"✓ Deleted {deleted_count} used clips from storage")

# -------------------------
# Build Playlist
# -------------------------

def build_playlist(imgs: List[str], seed: int, total_seconds: float, seg_min: int, seg_max: int) -> List[str]:
    rnd = random.Random(seed)
    avg_seg = (seg_min + seg_max) / 2
    need = int(total_seconds // avg_seg) + 5
    
    playlist = []
    pool = []
    last_img = None

    while len(playlist) < need:
        if not pool:
            pool = imgs[:]
            rnd.shuffle(pool)
            if last_img and pool[0] == last_img and len(pool) > 1:
                pool[0], pool[-1] = pool[-1], pool[0]
        
        img = pool.pop(0)
        playlist.append(img)
        last_img = img
        
    return playlist

def build_segment_plan(total_seconds: float, seed: int, seg_min: int, seg_max: int) -> List[float]:
    rnd = random.Random(seed + 999)
    plan = []
    s = 0.0
    while s < total_seconds:
        d = rnd.uniform(seg_min, seg_max)
        plan.append(d)
        s += d
    return plan

# -------------------------
# Math Generation for FFmpeg
# -------------------------

def generate_ken_burns_expr(
    duration: float,
    img_w: int,
    img_h: int,
    out_w: int,
    out_h: int,
    seed: int
) -> str:
    """
    Generates the 'zoompan' expression for a smooth Ken Burns effect.
    Uses smoothstep interpolation: p = t/dur; p = p*p*(3-2*p).
    """
    rnd = random.Random(seed)
    
    # Aspect ratios
    aspect_out = out_w / out_h
    aspect_img = img_w / img_h
    
    # Max crop size (fitting the aspect ratio)
    if aspect_img > aspect_out:
        h_crop_max = img_h
        w_crop_max = img_h * aspect_out
    else:
        w_crop_max = img_w
        h_crop_max = img_w / aspect_out
        
    # Zoom levels (relative to the max crop)
    # 1.0 = max crop
    # 2.0 = 2x zoom in (smaller crop, more dramatic)
    zoom_min = 1.0
    zoom_max = 2.0  # Increased from 1.6 for more dramatic motion
    
    z1 = rnd.uniform(zoom_min, zoom_max)
    z2 = rnd.uniform(zoom_min, zoom_max)
    
    # Crop sizes
    w1, h1 = w_crop_max / z1, h_crop_max / z1
    w2, h2 = w_crop_max / z2, h_crop_max / z2
    
    # Random centers
    def get_random_center(w_crop, h_crop):
        min_x = w_crop / 2
        max_x = img_w - w_crop / 2
        min_y = h_crop / 2
        max_y = img_h - h_crop / 2
        return rnd.uniform(min_x, max_x), rnd.uniform(min_y, max_y)

    x1, y1 = get_random_center(w1, h1)
    x2, y2 = get_random_center(w2, h2)
    
    # FFmpeg Expression Construction
    # Using SMOOTHERSTEP (polynomial degree 5) for ultra-smooth motion
    # This is smoother than smoothstep (degree 3)
    
    # Smootherstep formula (Ken Perlin):
    # p = time / duration
    # p = clamp(p, 0, 1)
    # s = p^3 * (p * (p * 6 - 15) + 10)
    # This gives C2 continuity (smooth acceleration AND jerk)
    
    # Interpolation:
    # val = v1 + (v2 - v1) * s
    
    # We need to define variables in the expression to keep it readable (or just inline it).
    # FFmpeg expressions support st(var, val) and ld(var).
    
    # Variables:
    # 0: duration
    # 1: p (linear progress)
    # 2: s (smooth progress)
    # 3: current width (w_curr)
    # 4: current height (h_curr)
    # 5: current center x (x_curr)
    # 6: current center y (y_curr)
    
    # We construct a single string for 'z', 'x', 'y'.
    # Since 'z' is evaluated first, we can compute shared vars there?
    # Actually, zoompan evaluates z, then x, then y.
    # We can compute everything in 'z' and store in variables, then load in x and y.
    
    # Constants
    dur = duration
    
    # Expression for Z (zoom factor)
    # FFmpeg zoom is relative to input size. 
    # Our math calculates crop size.
    # zoom = img_w / w_curr (assuming we crop width)
    # But wait, zoompan works by cropping a window of size (iw/zoom) x (ih/zoom).
    # So w_curr = img_w / zoom  => zoom = img_w / w_curr.
    
    # Let's build the expression string.
    # We use string formatting carefully.
    
    expr_setup = (
        f"st(0, {dur});"                  # store duration
        f"st(1, time/ld(0));"             # p = time / duration
        # Smootherstep: s = p^3 * (p * (p * 6 - 15) + 10)
        # Rewritten: s = p*p*p * (p*(p*6 - 15) + 10)
        f"st(2, ld(1)*ld(1)*ld(1)*(ld(1)*(ld(1)*6-15)+10));" # smootherstep
        
        # Interpolate dimensions and center
        f"st(3, {w1} + ({w2}-{w1})*ld(2));" # w_curr
        f"st(4, {h1} + ({h2}-{h1})*ld(2));" # h_curr
        f"st(5, {x1} + ({x2}-{x1})*ld(2));" # x_curr
        f"st(6, {y1} + ({y2}-{y1})*ld(2));" # y_curr
    )
    
    # Z expression: zoom = img_w / w_curr
    # Note: We must handle aspect ratio. 
    # If we crop w_curr, does it match h_curr? 
    # Our math ensures w_curr/h_curr = aspect_out.
    # zoompan assumes square pixels usually.
    # zoom = iw / w_curr
    
    expr_z = f"{expr_setup} {img_w}/ld(3)"
    
    # X expression: x_curr - w_curr/2
    expr_x = "ld(5) - ld(3)/2"
    
    # Y expression: y_curr - h_curr/2
    expr_y = "ld(6) - ld(4)/2"
    
    return expr_z, expr_x, expr_y

# -------------------------
# Filtergraph Builder
# -------------------------

def build_filtergraph(
    playlist: List[str],
    seg_plan: List[float],
    seed: int,
    width: int,
    height: int,
    fps: int,
    crossfade: float,
    overlays: List[str] = [],
) -> Tuple[str, List[str]]:
    rnd = random.Random(seed + 12345)
    inputs = []
    filters = []
    
    transitions = [
        'fade', 'wipeleft', 'wiperight', 'slideup', 'slidedown', 
        'circleopen', 'circleclose', 'rectcrop', 'distance', 
        'fadeblack', 'fadewhite', 'smoothleft', 'smoothright'
    ]

    # Track input indices
    # 0..N-1: Images
    # N..M: Overlays (we will add them as needed)
    
    input_idx = 0
    
    # Overlay pool logic
    overlay_pool = []
    last_overlay = None
    
    # Track image usage for smart flipping
    img_usage = {}
    
    for i, (img_path, seg_dur) in enumerate(zip(playlist, seg_plan)):
        total_dur = seg_dur + crossfade
        
        # 1. Image Input
        inputs.append(f'-loop 1 -t {total_dur:.3f} -i "{img_path}"')
        img_idx = input_idx
        input_idx += 1
        
        # 2. Overlay Input (Optional)
        use_overlay = False
        ov_idx = -1
        if overlays:
            # Smart shuffle logic
            if not overlay_pool:
                overlay_pool = overlays[:]
                rnd.shuffle(overlay_pool)
                # Avoid immediate repeat if possible
                if len(overlays) > 1 and last_overlay and overlay_pool[0] == last_overlay:
                    overlay_pool.append(overlay_pool.pop(0))
            
            ov_path = overlay_pool.pop(0)
            last_overlay = ov_path
            
            # Loop overlay to match duration
            inputs.append(f'-stream_loop -1 -t {total_dur:.3f} -i "{ov_path}"')
            ov_idx = input_idx
            input_idx += 1
            use_overlay = True
        
        # Get image dimensions
        try:
            img_arr = iio.imread(img_path)
            h_img, w_img = img_arr.shape[:2]
        except Exception:
            w_img, h_img = 1920, 1080

        # Generate Ken Burns Math
        clip_seed = seed + i
        z_expr, x_expr, y_expr = generate_ken_burns_expr(
            total_dur, w_img, h_img, width, height, clip_seed
        )
        
        # Smart Flip Logic
        # Flip every 2nd time the image appears to create variety
        usage_count = img_usage.get(img_path, 0)
        img_usage[img_path] = usage_count + 1
        
        do_flip = (usage_count % 2 != 0)
        flip_filter = "hflip," if do_flip else ""
        
        c = rnd.uniform(0.95, 1.1)
        b = rnd.uniform(-0.02, 0.02)
        s = rnd.uniform(0.9, 1.2)
        
        style_roll = rnd.random()
        if style_roll < 0.08:
            eq_filter = f"hue=s=0,eq=contrast=1.1:brightness=0.05,"
        elif style_roll < 0.15:
            eq_filter = "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,"
        else:
            eq_filter = f"eq=contrast={c:.2f}:brightness={b:.2f}:saturation={s:.2f},"

        # Construct Filter Chain for this segment
        # Step A: Process Image (Ken Burns + Color)
        # Note: zoompan outputs what it gets, or yuv420p.
        base_filter = (
            f'[{img_idx}:v]'
            f'{flip_filter}'
            f'{eq_filter}'
            f'format=yuv420p,'
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={int(total_dur*fps)}:s={width}x{height}:fps={fps}"
            f'[base{i}]'
        )
        filters.append(base_filter)
        
        final_node = f"base{i}"
        
        # Step B: Apply Overlay (if exists)
        if use_overlay:
            # FIX: Pink tint is caused by Screen blending on YUV. 
            # We must convert to RGB (gbrp) for blending, then back to YUV.
            
            # 1. Convert Base to RGB
            filters.append(f"[{final_node}]format=gbrp[base_rgb{i}]")
            
            # 2. Prepare Overlay (Scale -> RGB)
            ov_filter = (
                f'[{ov_idx}:v]scale={width}:{height}:force_original_aspect_ratio=increase,'
                f'crop={width}:{height},fps={fps},format=gbrp[ov{i}];'
                f'[base_rgb{i}][ov{i}]blend=all_mode=screen:shortest=1[blended_rgb{i}];'
                f'[blended_rgb{i}]format=yuv420p[blended{i}]'
            )
            filters.append(ov_filter)
            final_node = f"blended{i}"
        
        # Rename final node to v{i} for xfade chain
        # FIX: Force timebase (settb=AVTB) to prevent xfade errors
        filters.append(f"[{final_node}]settb=AVTB,fps={fps}[v{i}]")

    # Chain xfade
    xfade_chain = ""
    if len(seg_plan) == 1:
        xfade_chain = "[v0]null[vout]"
    else:
        acc = seg_plan[0]
        trans = rnd.choice(transitions)
        xfade_chain = f"[v0][v1]xfade=transition={trans}:duration={crossfade}:offset={acc-crossfade}[vx1]"
        for k in range(2, len(seg_plan)):
            acc += seg_plan[k-1]
            prev = f"vx{k-1}"
            cur = f"v{k}"
            out = f"vx{k}"
            trans = rnd.choice(transitions)
            xfade_chain += f";[{prev}][{cur}]xfade=transition={trans}:duration={crossfade}:offset={acc-crossfade}[{out}]"
        xfade_chain += f";[vx{len(seg_plan)-1}]null[vout]"

    # Post processing
    # If no overlays, add procedural noise/vignette
    if not overlays:
        post = (
            "[vout]"
            "noise=alls=10:allf=t+u,"
            "vignette=PI/4,"
            "eq=brightness=0.01:saturation=1.04"
            "[video]"
        )
    else:
        # If overlays exist, just slight EQ
        post = (
            "[vout]"
            "eq=brightness=0.01:saturation=1.04"
            "[video]"
        )

    filter_complex = ";".join(filters) + ";" + xfade_chain + ";" + post
    return filter_complex, inputs

# -------------------------
# Main Render
# -------------------------



def render_video(
    audio_path: str,
    images_dir: str,
    overlays_dir: str,
    out_path: str,
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    seg_min: int = 20,
    seg_max: int = 40,
    crossfade: float = 1.5,
):
    # 1. Setup
    total_seconds = get_audio_duration(audio_path)
    print(f"Audio duration: {total_seconds/60:.2f} minutes")
    
    imgs = list_images(images_dir)
    overlays = list_overlays(overlays_dir)
    print(f"Found {len(overlays)} overlay effects.")
    
    if seed == 0:
        seed = random.randint(0, 100000) + int(time.time())
    print(f"Seed: {seed}")
    
    playlist = build_playlist(imgs, seed, total_seconds, seg_min, seg_max)
    seg_plan = build_segment_plan(total_seconds, seed, seg_min, seg_max)
    
    n = min(len(playlist), len(seg_plan))
    playlist = playlist[:n]
    seg_plan = seg_plan[:n]
    
    print(f"Total clips: {n}")
    
    # -------------------------
    # CLIP STORAGE INTEGRATION
    # -------------------------
    storage_dir = "clip_storage"
    storage_clips, remaining_duration = get_clips_from_storage(total_seconds, storage_dir)
    
    used_clip_paths = []
    final_clip_files = []
    
    if storage_clips:
        # Use clips from storage
        print(f"Using {len(storage_clips)} clips from storage")
        for filepath, duration in storage_clips:
            final_clip_files.append(filepath)
            used_clip_paths.append(filepath)
    
    
    # Generate new clips if needed
    if remaining_duration > 0:
        print(f"Generating new clips for remaining {remaining_duration:.1f}s...")
        
        # Generate new clips for the missing duration
        # Build playlist and segment plan for the remaining duration only
        new_playlist = build_playlist(imgs, seed + 1000, remaining_duration, seg_min, seg_max)
        new_seg_plan = build_segment_plan(remaining_duration, seed + 1000, seg_min, seg_max)
        
        n_new = min(len(new_playlist), len(new_seg_plan))
        new_playlist = new_playlist[:n_new]
        new_seg_plan = new_seg_plan[:n_new]
        
        print(f"Generating {n_new} new clips...")
        
        # Render new clips to a temporary file
        temp_new_video = str(Path(out_path).with_suffix(".new.mp4"))
        _render_single_video(
            new_playlist, new_seg_plan, seed + 1000, width, height, fps, crossfade,
            overlays, temp_new_video, remaining_duration
        )
        
        # Concatenate storage clips + new clips
        concat_list_file = "hybrid_concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            # First, add storage clips
            for clip_path in final_clip_files:
                abs_path = Path(clip_path).resolve().as_posix()
                f.write(f"file '{abs_path}'\n")
            # Then, add new generated video
            abs_new_path = Path(temp_new_video).resolve().as_posix()
            f.write(f"file '{abs_new_path}'\n")
        
        temp_concat = str(Path(out_path).with_suffix(".concat.mp4"))
        cmd_concat = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-f concat -safe 0 -i "{concat_list_file}" '
            f'-c copy "{temp_concat}"'
        )
        print(">> Concatenating storage clips + new clips...")
        run(cmd_concat)
        
        # Add audio
        cmd_mux = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-i "{temp_concat}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'-shortest -movflags +faststart '
            f'"{out_path}"'
        )
        print(">> Adding audio...")
        run(cmd_mux)
        
        # Cleanup
        try:
            os.remove(temp_new_video)
            os.remove(temp_concat)
            os.remove(concat_list_file)
        except OSError:
            pass
        
        # Delete used clips from storage
        delete_used_clips(used_clip_paths, storage_dir)
        
        print("Done:", out_path)
        return  # Exit early
        
    else:
        # We have enough clips from storage, use them directly
        print("✓ All clips from storage, skipping generation")
        
        # Concatenate storage clips
        concat_list_file = "storage_concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for clip_path in final_clip_files:
                # Convert to absolute path and use forward slashes for FFmpeg
                abs_path = Path(clip_path).resolve().as_posix()
                f.write(f"file '{abs_path}'\n")
        
        temp_video = str(Path(out_path).with_suffix(".concat.mp4"))
        cmd_concat = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-f concat -safe 0 -i "{concat_list_file}" '
            f'-c copy "{temp_video}"'
        )
        print(">> Concatenating storage clips...")
        run(cmd_concat)
        
        # Add audio
        cmd_mux = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-i "{temp_video}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'-shortest -movflags +faststart '
            f'"{out_path}"'
        )
        print(">> Adding audio...")
        run(cmd_mux)
        
        # Cleanup
        try:
            os.remove(temp_video)
            os.remove(concat_list_file)
        except OSError:
            pass
        
        # Delete used clips from storage
        delete_used_clips(used_clip_paths, storage_dir)
        
        print("Done:", out_path)
        return  # Exit early, no need for normal rendering
    
    # -------------------------
    # NORMAL RENDERING (if storage wasn't enough)
    # -------------------------
    
    # FIX: Split into chunks to avoid FFmpeg filter complexity limits
    # Reduced to 30 for better RAM efficiency
    MAX_CLIPS_PER_CHUNK = 30
    
    if n <= MAX_CLIPS_PER_CHUNK:
        # Small video, render directly
        temp_bg = str(Path(out_path).with_suffix(".bg.mp4"))
        _render_single_video(
            playlist, seg_plan, seed, width, height, fps, crossfade,
            overlays, temp_bg, total_seconds
        )
        
        # Add audio
        cmd_mux = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-i "{temp_bg}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'-shortest -movflags +faststart '
            f'"{out_path}"'
        )
        print("Muxing audio...")
        run(cmd_mux)
        
        # Cleanup
        try:
            os.remove(temp_bg)
        except OSError:
            pass
        
        print("Done:", out_path)
    else:
        # Large video, split into chunks
        print(f"Video too large ({n} clips), splitting into chunks of {MAX_CLIPS_PER_CHUNK}...")
        
        chunk_files = []
        chunk_start = 0
        chunk_idx = 0
        
        while chunk_start < n:
            chunk_end = min(chunk_start + MAX_CLIPS_PER_CHUNK, n)
            chunk_playlist = playlist[chunk_start:chunk_end]
            chunk_seg_plan = seg_plan[chunk_start:chunk_end]
            chunk_duration = sum(chunk_seg_plan)
            
            chunk_file = str(Path(out_path).with_suffix(f".chunk{chunk_idx}.mp4"))
            
            print(f"\nRendering chunk {chunk_idx + 1}/{(n + MAX_CLIPS_PER_CHUNK - 1) // MAX_CLIPS_PER_CHUNK} ({chunk_end - chunk_start} clips, {chunk_duration/60:.1f} min)...")
            
            _render_single_video(
                chunk_playlist, chunk_seg_plan, seed + chunk_idx * 1000,
                width, height, fps, crossfade, overlays, chunk_file, chunk_duration
            )
            
            chunk_files.append(chunk_file)
            chunk_start = chunk_end
            chunk_idx += 1
        
        # Concatenate chunks
        print(f"\nConcatenating {len(chunk_files)} chunks...")
        concat_list_file = "concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")
        
        temp_video = str(Path(out_path).with_suffix(".concat.mp4"))
        cmd_concat = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-f concat -safe 0 -i "{concat_list_file}" '
            f'-c copy "{temp_video}"'
        )
        print(">> Concatenating chunks...")
        run(cmd_concat)
        
        # Add audio
        cmd_mux = (
            f'ffmpeg -y -hide_banner -loglevel warning '
            f'-i "{temp_video}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -b:a 192k '
            f'-shortest -movflags +faststart '
            f'"{out_path}"'
        )
        print(">> Adding audio...")
        run(cmd_mux)
        
        # Cleanup
        try:
            os.remove(temp_video)
            os.remove(concat_list_file)
            for chunk_file in chunk_files:
                os.remove(chunk_file)
        except OSError:
            pass
        
        print("Done:", out_path)

def _render_single_video(
    playlist, seg_plan, seed, width, height, fps, crossfade,
    overlays, out_path, total_seconds
):
    """Render a single video segment (used for both full videos and chunks)"""
    
    print(f"Generating filtergraph for {len(playlist)} clips...")
    filter_complex, input_args = build_filtergraph(
        playlist, seg_plan, seed, width, height, fps, crossfade, overlays
    )
    
    # 2. Write Filter Script
    filter_script_path = "filter_script.txt"
    with open(filter_script_path, "w", encoding="utf-8") as f:
        f.write(filter_complex)
        
    # 3. Render
    temp_bg = str(Path(out_path).with_suffix(".bg.mp4"))
    
    inputs_str = " ".join(input_args)
    
    # Build batch file
    batch_file_path = "render_video.bat"
    
    # Build the full FFmpeg command as a string
    cmd_parts = ["ffmpeg"]
    cmd_parts.extend(["-y", "-hide_banner", "-loglevel", "error", "-stats"])
    
    # Add inputs
    input_tokens = shlex.split(inputs_str)
    cmd_parts.extend(input_tokens)
    
    # Filter complex
    cmd_parts.extend(["-filter_complex_script", filter_script_path])
    
    # Output options
    cmd_parts.extend(["-map", "[video]"])
    cmd_parts.extend(["-r", str(fps)])
    cmd_parts.extend(["-t", f"{total_seconds:.3f}"])
    
    # GPU encoding with speed optimization and fixed bitrate
    cmd_parts.extend(["-c:v", "h264_nvenc"])
    cmd_parts.extend(["-preset", "p4"])         # p4 = faster than p2, same quality
    cmd_parts.extend(["-tune", "hq"])           # High quality tuning
    cmd_parts.extend(["-b:v", "1100k"])         # Fixed bitrate 1500 kbps
    cmd_parts.extend(["-maxrate", "1100k"])     # Max bitrate
    cmd_parts.extend(["-bufsize", "3000k"])     # Buffer size (2x bitrate)
    cmd_parts.extend(["-2pass", "0"])           # Disable 2-pass for speed
    cmd_parts.extend(["-rc-lookahead", "20"])   # Reduce GPU memory
    cmd_parts.extend(["-surfaces", "8"])        # Limit NVENC surfaces
    cmd_parts.extend(["-threads", "4"])         # Limit CPU threads
    cmd_parts.extend(["-max_muxing_queue_size", "1024"])  # Limit buffer
    cmd_parts.extend(["-pix_fmt", "yuv420p"])
    cmd_parts.append(f'"{temp_bg}"')
    
    # Write to batch file
    with open(batch_file_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        # Join arguments, properly quoting those with spaces
        cmd_line = " ".join(
            f'"{arg}"' if (" " in arg and not arg.startswith('"')) else arg
            for arg in cmd_parts
        )
        f.write(cmd_line + "\n")
    
    print("Rendering video (GPU Accelerated)...")
    print(f">> ffmpeg with {len(cmd_parts)} arguments (via batch file)")
    
    # Execute batch file
    process = subprocess.Popen(
        batch_file_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace',
        shell=True
    )
    
    # Progress tracking
    time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
    speed_pattern = re.compile(r"speed=\s*(\d+\.?\d*)x")
    
    print("\nRendering Progress:")
    print("-" * 60)
    
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
                
            t_match = time_pattern.search(line)
            s_match = speed_pattern.search(line)
            
            if t_match:
                h, m, s = map(float, t_match.groups())
                current_seconds = h * 3600 + m * 60 + s
                
                percent = min(100, (current_seconds / total_seconds) * 100)
                
                speed = "N/A"
                eta = "N/A"
                
                if s_match:
                    speed_val = float(s_match.group(1))
                    speed = f"{speed_val:.2f}x"
                    if speed_val > 0:
                        remaining = (total_seconds - current_seconds) / speed_val
                        eta = f"{int(remaining)}s"
                
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Speed: {speed} | ETA: {eta}   ")
                sys.stdout.flush()
            
            elif "Error" in line or "failed" in line:
                print(f"\n{line}")

    except Exception as e:
        print(f"\nError reading progress: {e}")
        
    process.wait()
    print("\n" + "-" * 60)
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}")
    
    # 4. Mux Audio (only for full videos, not chunks)
    if not out_path.endswith((".chunk0.mp4", ".chunk1.mp4", ".chunk2.mp4", ".chunk3.mp4", ".chunk4.mp4", ".chunk5.mp4", ".chunk6.mp4", ".chunk7.mp4", ".chunk8.mp4", ".chunk9.mp4")):
        # This is a full video, add audio
        # Actually, for chunks we skip audio, we'll add it at the end
        pass
    
    # Just rename temp to final for chunks (no audio)
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(temp_bg, out_path)
        os.remove(filter_script_path)
        os.remove(batch_file_path)
    except OSError as e:
        print(f"Cleanup warning: {e}")

def main():
    start = time.time()

    # ====== INPUT ======
    AUDIO_PATH  = r"E:/work/work/AudiobookBay/Danielle Garrett - Nine Lives Magic/Danielle Garrett - Nine Lives Magic 02 - Hexed Hiss-tory/output v2/mp3 part 4.MP3"
    # AUDIO_PATH  = r"files/quyenyeuduoi.mp3"
    IMAGES_DIR  = r"images"
    OVERLAYS_DIR = r"overlays"
    OUTPUT_PATH = r"E:/work/work/AudiobookBay/Danielle Garrett - Nine Lives Magic/Danielle Garrett - Nine Lives Magic 02 - Hexed Hiss-tory/output v2/video_stock_part4.mp4"
    # OUTPUT_PATH = r"outputs/quyenyeuduoi.mp4"

    # ====== CONFIG ======
    WIDTH  = 1280
    HEIGHT = 720
    FPS    = 24     # Cinema standard (faster than 30, imperceptible difference)
    
    # Motion Config
    SEED       = 0
    SEG_MIN    = 90     # Longer segments = fewer transitions = faster render
    SEG_MAX    = 150
    CROSSFADE  = 1.5

    # render_video(
    #     audio_path=AUDIO_PATH,
    #     images_dir=IMAGES_DIR,
    #     overlays_dir=OVERLAYS_DIR,
    #     out_path=OUTPUT_PATH,
    #     seed=SEED,
    #     width=WIDTH,
    #     height=HEIGHT,
    #     fps=FPS,
    #     seg_min=SEG_MIN,
    #     seg_max=SEG_MAX,
    #     crossfade=CROSSFADE,
    # )
    
    generate_clip_storage(
        count=500,
        storage_dir="clip_storage",
        images_dir="images",
        overlays_dir="overlays",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        seg_min=SEG_MIN,
        seg_max=SEG_MAX,
        seed=SEED,
        max_workers=1 
    )
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
if __name__ == "__main__":
    main()
