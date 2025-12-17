# Faster Whisper GUI

A professional Gradio-based GUI for faster-whisper audio/video transcription with CUDA acceleration and multiprocessing architecture.

## Features

- ✅ **CUDA Acceleration** - GPU-accelerated transcription (with CPU fallback)
- ✅ **Multiprocessing Architecture** - Solves Gradio threading conflicts with CUDA
- ✅ **Video Support** - Automatic video-to-audio conversion using FFmpeg
- ✅ **Multiple Formats** - Export to TXT, JSON, SRT, and VTT subtitle formats
- ✅ **Batch Processing** - Process multiple files with batched inference
- ✅ **Smart Organization** - Automatic file organization with subfolders
- ✅ **Real-time Progress** - Live progress tracking and detailed logs
- ✅ **SOLID Architecture** - Clean, modular, and maintainable code

## Requirements

- Python 3.12+
- FFmpeg (required for video conversion)
- CUDA-capable GPU (optional, for acceleration)

## Installation

1. **Install dependencies:**

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

2. **Install FFmpeg:**

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

3. **For CUDA support (optional):**

Install CUDA drivers and libraries as per [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper#gpu)

## Usage

### Launch the GUI

```bash
python main.py
```

Or with uv:

```bash
uv run main.py
```

The GUI will be available at `http://localhost:7860`

### Default Configuration

The GUI comes with optimized default settings:

- **Model**: large-v3 (best accuracy)
- **Device**: CUDA (GPU acceleration)
- **Compute Type**: int8_float16 (balanced speed/accuracy)
- **Beam Size**: 5 (optimal quality)
- **Batch Size**: 8 (efficient processing)
- **VAD Filter**: Enabled (removes silence)
- **Word Timestamps**: Disabled (faster)
- **Batched Inference**: Enabled (faster for long files)
- **Output Formats**: TXT, JSON, SRT, VTT

### Workflow

1. **Upload Files** - Select audio/video files from `input/` folder
2. **Configure Settings** (optional - defaults are optimized):
   - Model: tiny, base, small, medium, large-v3, turbo
   - Device: CUDA or CPU
   - Language: Auto-detect or specify
   - Advanced options: beam size, batch size, VAD filter
3. **Start Transcription** - Click "🚀 Start Transcription"
4. **View Results** - See transcripts in real-time in multiple formats
5. **Download** - Files are organized in `output/` folder with subfolders

### File Organization

```
output/
├── video_name_YYYYMMDD_HHMMSS/
│   ├── transcript.txt
│   ├── transcript.json
│   ├── transcript.srt
│   ├── transcript.vtt
│   └── video_name.mp4  (moved after processing)
```

## Architecture

### Multiprocessing Design

The application uses a **multiprocessing architecture** to solve critical CUDA threading conflicts with Gradio:

**Problem**: Gradio's threading model conflicts with CUDA context initialization, causing `transcribe()` calls to hang indefinitely.

**Solution**: Transcription runs in a **separate process** with isolated CUDA context:

```
Main Process (GUI)              Worker Process
├── File upload                 ├── CUDA initialization
├── Video conversion            ├── Model loading
├── Progress monitoring    ←──→ ├── Transcription
├── Result display              └── Return segments
└── Temp file cleanup
```

### SOLID Principles

Following clean architecture with single responsibility:

```
src/
├── environment_checker.py   # Environment validation
├── media_converter.py        # FFmpeg video-to-audio (main process)
├── transcription_service.py  # Whisper model wrapper (worker process)
├── file_manager.py           # File operations & organization
├── output_formatter.py       # TXT/JSON/SRT/VTT generation
├── config.py                 # Configuration & settings
└── gui.py                    # Gradio interface + multiprocessing
```

**Key Design Decisions**:

- Video conversion happens in **main process** (UI responsibility)
- Transcription happens in **worker process** (isolated CUDA context)
- Clean separation via Queue-based IPC

## Technical Details

Default settings (can be modified in `src/config.py`):

- **Device**: CUDA (falls back to CPU if unavailable)
- **Compute Type**: float16 (CUDA) / int8 (CPU)
- **Model**: large-v3
- **Beam Size**: 5
- **VAD Filter**: Enabled
- **Audio Format**: WAV (16kHz, mono)

## Environment Validation

The application performs pre-flight checks:

1. ✅ FFmpeg availability (required)
2. ✅ CUDA availability (recommended)
3. ✅ Python dependencies
4. ✅ Device selection validation

If FFmpeg is missing, the application will exit with installation instructions.
If CUDA is selected but unavailable, transcription will be blocked.

## Supported Formats

**Audio**: MP3, WAV, M4A, FLAC, AAC, OGG, OPUS, WMA

**Video**: MP4, MKV, AVI, MOV, WMV, FLV, WEBM, M4V, MPG, MPEG

## Troubleshooting

### FFmpeg not found

```
Error: FFmpeg not found in PATH
Solution: Install FFmpeg and ensure it's in your system PATH
```

### CUDA not available

```
Warning: CUDA not available
Solution: Install CUDA drivers or use CPU mode
```

### Model download issues

```
Error: Cannot download model
Solution: Check internet connection, models are downloaded on first use
```

## License

This project uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) under the MIT License.

## Credits

- **faster-whisper** by SYSTRAN
- **Whisper** by OpenAI
- **Gradio** for the web interface
- **FFmpeg** for media conversion
