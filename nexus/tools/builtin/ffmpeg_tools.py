"""
FFmpeg and media processing tools.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ...config import BASE_DIR
from ..registry import tool


SANDBOX = BASE_DIR / "data" / "workspace"
MEDIA_DIR = SANDBOX / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def _media_path(rel: str) -> Path:
    """Resolve a media file path safely within the MEDIA_DIR sandbox.
    Supports both relative paths (scoped to MEDIA_DIR) and absolute paths
    (checked against MEDIA_DIR for sandbox compliance).
    """
    p = Path(rel)
    sandbox = MEDIA_DIR.resolve()
    if p.is_absolute():
        p = p.resolve()
    else:
        p = (MEDIA_DIR / rel).resolve()
    if sandbox not in p.parents and p != sandbox:
        raise ValueError(f"Path escape attempt: {rel} — path must be inside {MEDIA_DIR}")
    return p


@tool(
    name="ffmpeg_convert",
    description="Convert media files between formats using FFmpeg (video/audio/image). "
                "Supports: mp4, avi, mov, mkv, webm, mp3, wav, flac, ogg, gif, etc.",
    parameters_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input file path"},
            "output": {"type": "string", "description": "Output file path"},
            "format": {"type": "string", "description": "Target format (mp4, mp3, gif, etc.)"},
            "codec": {"type": "string", "description": "Video codec (libx264, h264_nvenc, etc.)"},
            "bitrate": {"type": "string", "description": "Video bitrate (1M, 500k, etc.)"},
            "resolution": {"type": "string", "description": "Scale (1920x1080, 1280x720, etc.)"},
            "fps": {"type": "integer", "description": "Frame rate (24, 30, 60)"},
            "audio_codec": {"type": "string", "description": "Audio codec (aac, libmp3lame, etc.)"},
            "audio_bitrate": {"type": "string", "description": "Audio bitrate (192k, 128k)"},
            "format_options": {"type": "string", "description": "Extra FFmpeg options"},
        },
        "required": ["input", "output"],
    },
    category="media",
)
async def ffmpeg_convert(params: dict) -> str:
    """Convert media between formats."""
    import shutil
    if not shutil.which("ffmpeg"):
        return "Error: FFmpeg not installed. Install with: brew install ffmpeg (mac) or apt install ffmpeg (linux)"
    
    try:
        input_path = _media_path(params["input"])
    except ValueError as e:
        return f"Error: {e}"
    if not input_path.exists():
        return f"Error: Input file not found: {input_path}"
    
    try:
        output_path = _media_path(params["output"])
    except ValueError as e:
        return f"Error: {e}"
    
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    
    # Video options
    if params.get("codec"):
        cmd.extend(["-c:v", params["codec"]])
    if params.get("bitrate"):
        cmd.extend(["-b:v", params["bitrate"]])
    if params.get("resolution"):
        cmd.extend(["-s", params["resolution"]])
    if params.get("fps"):
        cmd.extend(["-r", str(params["fps"])])
    
    # Audio options
    if params.get("audio_codec"):
        cmd.extend(["-c:a", params["audio_codec"]])
    if params.get("audio_bitrate"):
        cmd.extend(["-b:a", params["audio_bitrate"]])
    
    if params.get("format_options"):
        cmd.extend(params["format_options"].split())
    
    cmd.append(str(output_path))
    
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=300)
        if result.returncode != 0:
            stderr = result.stderr or ""
            return f"FFmpeg error: {stderr}"
        return f"Converted: {input_path.name} → {output_path.name} ({output_path.stat().st_size:,} bytes)"
    except subprocess.TimeoutExpired:
        return "Error: Conversion timed out after 5 minutes"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ffmpeg_extract_audio",
    description="Extract audio track from video file.",
    parameters_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input video file"},
            "output": {"type": "string", "description": "Output audio file"},
            "format": {"type": "string", "default": "mp3", "description": "Audio format (mp3, wav, flac, ogg)"},
            "bitrate": {"type": "string", "default": "192k", "description": "Audio bitrate"},
        },
        "required": ["input", "output"],
    },
    category="media",
)
async def ffmpeg_extract_audio(params: dict) -> str:
    """Extract audio from video."""
    import shutil
    if not shutil.which("ffmpeg"):
        return "Error: FFmpeg not installed"
    
    try:
        input_path = _media_path(params["input"])
    except ValueError as e:
        return f"Error: {e}"
    if not input_path.exists():
        return f"Error: Input not found: {input_path}"
    
    try:
        output_path = _media_path(params["output"])
    except ValueError as e:
        return f"Error: {e}"
    fmt = params.get("format", "mp3")
    bitrate = params.get("bitrate", "192k")
    
    codec_map = {"mp3": "libmp3lame", "wav": "pcm_s16le", "flac": "flac", "ogg": "libvorbis"}
    audio_codec = codec_map.get(fmt, "libmp3lame")
    
    cmd = ["ffmpeg", "-y", "-i", str(input_path), 
          "-vn", "-c:a", audio_codec, "-b:a", bitrate, str(output_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        if result.returncode != 0:
            stderr = result.stderr or ""
            return f"Error: {stderr}"
        return f"Extracted audio: {output_path.name} ({output_path.stat().st_size:,} bytes)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ffmpeg_create_gif",
    description="Create animated GIF from video or images with optimizations.",
    parameters_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input video file"},
            "output": {"type": "string", "description": "Output GIF file"},
            "start": {"type": "number", "description": "Start time in seconds"},
            "duration": {"type": "number", "description": "Duration in seconds"},
            "fps": {"type": "integer", "default": 15, "description": "Output FPS"},
            "width": {"type": "integer", "description": "Width (auto if not set)"},
            "palette": {"type": "boolean", "default": True, "description": "Use palette for better quality"},
        },
        "required": ["input", "output"],
    },
    category="media",
)
async def ffmpeg_create_gif(params: dict) -> str:
    """Create optimized GIF from video."""
    import shutil
    if not shutil.which("ffmpeg"):
        return "Error: FFmpeg not installed"
    
    try:
        input_path = _media_path(params["input"])
    except ValueError as e:
        return f"Error: {e}"
    try:
        output_path = _media_path(params["output"])
    except ValueError as e:
        return f"Error: {e}"
    
    if not input_path.exists():
        return f"Error: Input not found"
    
    cmd = ["ffmpeg", "-y"]
    
    if params.get("start", 0) > 0:
        cmd.extend(["-ss", str(params["start"])])
    if params.get("duration"):
        cmd.extend(["-t", str(params["duration"])])
    
    cmd.extend(["-i", str(input_path)])
    
    # Palette generation for better quality
    if params.get("palette", True):
        palette_cmd = ["ffmpeg", "-y", "-ss", str(params.get("start", 0)),
                      "-t", str(params.get("duration", 5)),
                      "-i", str(input_path),
                      "-vf", f"fps={params.get('fps',15)},scale={params.get('width',480)}:-1:flags=lanczos,palettegen",
                      "-y", str(output_path.with_suffix(".palette.png"))]
        subprocess.run(palette_cmd, capture_output=True)
        
        filter_vf = f"fps={params.get('fps',15)},scale={params.get('width',480)}:-1:flags=lanczos[x];[x][1:v]paletteuse"
        cmd = ["ffmpeg", "-y"]
        if params.get("start", 0) > 0:
            cmd.extend(["-ss", str(params["start"])])
        if params.get("duration"):
            cmd.extend(["-t", str(params["duration"])])
        cmd.extend(["-i", str(input_path), "-i", str(output_path.with_suffix(".palette.png")),
                  "-filter_complex", filter_vf, str(output_path)])
    else:
        fps = params.get("fps", 15)
        width = params.get("width", 480)
        cmd.extend(["-vf", f"fps={fps},scale={width}:-1:flags=lanczos", str(output_path)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        if result.returncode != 0:
            stderr = result.stderr or ""
            return f"Error: {stderr}"
        
        # Cleanup palette
        if output_path.with_suffix(".palette.png").exists():
            output_path.with_suffix(".palette.png").unlink()
        
        return f"Created GIF: {output_path.name} ({output_path.stat().st_size:,} bytes)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ffmpeg_merge_video",
    description="Concatenate multiple video files into one.",
    parameters_schema={
        "type": "object",
        "properties": {
            "inputs": {"type": "array", "items": {"type": "string"}, 
                     "description": "List of input video files"},
            "output": {"type": "string", "description": "Output merged file"},
            "format": {"type": "string", "default": "mp4", "description": "Output format"},
        },
        "required": ["inputs", "output"],
    },
    category="media",
)
async def ffmpeg_merge_video(params: dict) -> str:
    """Merge/join multiple video files."""
    import shutil
    if not shutil.which("ffmpeg"):
        return "Error: FFmpeg not installed"
    
    try:
        inputs = [_media_path(f) for f in params["inputs"]]
    except ValueError as e:
        return f"Error: {e}"
    try:
        output = _media_path(params["output"])
    except ValueError as e:
        return f"Error: {e}"
    
    # Check all exist
    missing = [f.name for f in inputs if not f.exists()]
    if missing:
        return f"Error: Missing files: {missing}"
    
    # Create concat list file
    list_file = MEDIA_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for inp in inputs:
            f.write(f"file '{inp.resolve()}'\n")
    
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
          "-c", "copy", str(output)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=300)
        list_file.unlink()
        
        if result.returncode != 0:
            stderr = result.stderr or ""
            return f"Error: {stderr}"
        return f"Merged {len(inputs)} videos → {output.name} ({output.stat().st_size:,} bytes)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ffmpeg_trim",
    description="Trim video to specific start/end times.",
    parameters_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input video"},
            "output": {"type": "string", "description": "Output video"},
            "start": {"type": "number", "description": "Start time in seconds"},
            "end": {"type": "number", "description": "End time in seconds"},
        },
        "required": ["input", "output", "start"],
    },
    category="media",
)
async def ffmpeg_trim(params: dict) -> str:
    """Trim video to specific time range."""
    import shutil
    if not shutil.which("ffmpeg"):
        return "Error: FFmpeg not installed"
    
    try:
        input_path = _media_path(params["input"])
    except ValueError as e:
        return f"Error: {e}"
    try:
        output_path = _media_path(params["output"])
    except ValueError as e:
        return f"Error: {e}"
    
    if not input_path.exists():
        return "Error: Input not found"
    
    start = params["start"]
    duration = params.get("end", 60) - start if params.get("end") else params.get("duration", 10)
    
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
          "-i", str(input_path), "-c", "copy", str(output_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        if result.returncode != 0:
            stderr = result.stderr or ""
            return f"Error: {stderr}"
        return f"Trimmed: {output_path.name} ({duration:.1f}s, {output_path.stat().st_size:,} bytes)"
    except Exception as e:
        return f"Error: {e}"