"""Gradio GUI for faster-whisper transcription."""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import shutil
from multiprocessing import Process, Queue
import time

from .environment_checker import EnvironmentChecker
from .transcription_service import TranscriptionService
from .media_converter import MediaConverter
from .file_manager import FileManager
from .output_formatter import OutputFormatter
from .config import (
    ModelConfig, TranscriptionOptions, ConversionOptions,
    FileOrganizationOptions, ModelSize, DeviceType, ComputeType,
    SUPPORTED_LANGUAGES, SUPPORTED_FORMATS
)

logger = logging.getLogger(__name__)


class WhisperGUI:
    """Gradio-based GUI for faster-whisper transcription."""
    
    def __init__(self):
        """Initialize WhisperGUI with all components."""
        self.env_checker = EnvironmentChecker()
        self.transcription_service: Optional[TranscriptionService] = None
        self.media_converter = MediaConverter()
        self.file_manager = FileManager()
        self.output_formatter = OutputFormatter()
        
        # State
        self.uploaded_files: List[str] = []
        self.current_file: Optional[str] = None
        self.current_output_dir: Optional[str] = None
        
        logger.info("WhisperGUI initialized")
    
    def validate_environment(self) -> Tuple[bool, str]:
        """
        Run environment validation.
        
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        return self.env_checker.validate_environment(
            require_cuda=False,  # Don't require CUDA at startup
            require_ffmpeg=True
        )
    
    def upload_files(self, files) -> Tuple[str, gr.Radio, str]:
        """
        Handle file upload and copy to input folder.
        
        Args:
            files: Uploaded file from Gradio (can be a string path or file object)
            
        Returns:
            Tuple[str, gr.Radio, str]: (status_message, updated_radio_with_selection, file_info)
        """
        if not files:
            return "No files uploaded", gr.Radio(choices=[]), ""
        
        self.uploaded_files = []
        valid_files = []
        invalid_files = []
        
        # Handle single file upload - Gradio returns the file path directly
        file_path = files if isinstance(files, str) else (files.name if hasattr(files, 'name') else str(files))
        
        # Copy to input folder
        success, new_path, msg = self.file_manager.copy_to_input_folder(file_path)
        
        if success:
            # Validate the copied file
            is_valid, validation_msg = self.file_manager.validate_audio_file(new_path)
            if is_valid:
                self.uploaded_files.append(new_path)
                valid_files.append(Path(new_path).name)
                # Auto-select the uploaded file - set the full path
                self.current_file = new_path
                logger.info(f"Auto-selected file: {self.current_file}")
            else:
                invalid_files.append(f"{Path(new_path).name}: {validation_msg}")
        else:
            invalid_files.append(f"{Path(file_path).name}: {msg}")
        
        status = f"{len(valid_files)} file(s) uploaded"
        if invalid_files:
            status += f" | {len(invalid_files)} failed"
        
        # Return updated radio with choices and selected value
        selected_file = valid_files[0] if valid_files else None
        all_files = self.get_input_files_list()
        
        # Generate file info
        file_info = ""
        if selected_file and self.current_file:
            file_path_obj = Path(self.current_file)
            size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            file_info = f"Selected: {selected_file} ({size_mb:.2f} MB)"
        
        return status, gr.Radio(choices=all_files, value=selected_file), file_info
    
    def get_input_files_list(self) -> List[str]:
        """
        Get list of files in input folder.
        
        Returns:
            List[str]: List of filenames in input folder
        """
        try:
            input_dir = Path(self.file_manager.options.input_dir)
            if not input_dir.exists():
                return []
            
            files = []
            for file in input_dir.iterdir():
                if file.is_file():
                    is_valid, _ = self.file_manager.validate_audio_file(str(file))
                    if is_valid:
                        files.append(file.name)
            
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing input files: {e}")
            return []
    
    def select_input_file(self, filename: str) -> str:
        """
        Select a file from input folder for transcription.
        
        Args:
            filename: Name of file in input folder
            
        Returns:
            str: Status message
        """
        if not filename:
            self.current_file = None
            logger.info("No file selected (filename is empty)")
            return "No file selected"
        
        try:
            input_dir = Path(self.file_manager.options.input_dir)
            file_path = input_dir / filename
            
            if file_path.exists():
                self.current_file = str(file_path)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"File selected: {self.current_file}")
                return f"Selected: {filename} ({size_mb:.2f} MB)"
            else:
                self.current_file = None
                logger.warning(f"File not found: {file_path}")
                return f"File not found: {filename}"
                
        except Exception as e:
            self.current_file = None
            logger.error(f"Error selecting file: {e}")
            return f"Error: {str(e)}"
    
    def delete_file(self, file_index: int) -> Tuple[str, str]:
        """Delete a file from upload list."""
        if 0 <= file_index < len(self.uploaded_files):
            removed = self.uploaded_files.pop(file_index)
            return self.get_file_list_html(), f"Removed: {Path(removed).name}"
        return self.get_file_list_html(), "Invalid file index"
    
    def get_file_list_html(self) -> str:
        """Get HTML representation of uploaded files."""
        if not self.uploaded_files:
            return "<div>No files uploaded</div>"
        
        html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for i, file in enumerate(self.uploaded_files, 1):
            html += f"<div style='padding: 5px; border-bottom: 1px solid #ddd;'>"
            html += f"<strong>{i}.</strong> {Path(file).name}</div>"
        html += "</div>"
        return html
    
    def move_files_back(self, output_folder_name: str) -> str:
        """
        Move processed files back from output folder to input directory.
        
        Args:
            output_folder_name: Name of output folder
            
        Returns:
            str: Status message
        """
        if not output_folder_name:
            return "No output folder selected"
        
        try:
            output_base = Path(self.file_manager.options.output_base_dir)
            data_dir = output_base / output_folder_name / "data"
            input_dir = Path("input")  # Default input directory
            input_dir.mkdir(exist_ok=True)
            
            if not data_dir.exists():
                return "Data folder doesn't exist"
            
            moved_files = []
            for file in data_dir.iterdir():
                if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
                    success, new_path, msg = self.file_manager.move_back_to_input(
                        str(file),
                        str(input_dir)
                    )
                    if success:
                        moved_files.append(file.name)
            
            if moved_files:
                return f"Moved {len(moved_files)} file(s) back to input folder: {', '.join(moved_files)}"
            else:
                return "No media files found to move back"
                
        except Exception as e:
            logger.error(f"Error moving files back: {e}")
            return f"Error: {str(e)}"
    
    def download_output_folder(self, output_folder_name: str) -> Tuple[Optional[str], dict, dict]:
        """
        Return the ZIP archive path if it exists and visibility updates.
        
        Args:
            output_folder_name: Name of output folder
            
        Returns:
            Tuple[Optional[str], dict, dict]: (archive_path, download_file_visibility, move_btn_visibility)
        """
        if not output_folder_name:
            return None, gr.update(visible=False), gr.update(visible=False)
        
        try:
            output_base = Path(self.file_manager.options.output_base_dir)
            folder_path = output_base / output_folder_name
            # ZIP is inside the folder now
            archive_path = folder_path / f"{output_folder_name}.zip"
            
            if not folder_path.exists():
                return None, gr.update(visible=False), gr.update(visible=False)
            
            if archive_path.exists():
                abs_path = str(archive_path.resolve())
                logger.info(f"Archive found at: {abs_path}")
                return abs_path, gr.update(visible=True), gr.update(visible=True)
            else:
                return None, gr.update(visible=False), gr.update(visible=True)
                
        except Exception as e:
            logger.error(f"Error getting archive: {e}")
            return None, gr.update(visible=False), gr.update(visible=False)
    
    def load_output_preview(self, output_folder_name: str) -> Tuple[str, str, str, str]:
        """
        Load output files from selected folder for preview.
        
        Args:
            output_folder_name: Name of output folder
            
        Returns:
            Tuple[str, str, str, str]: (txt_content, json_content, srt_content, vtt_content)
        """
        if not output_folder_name:
            return "", "", "", ""
        
        try:
            output_base = Path(self.file_manager.options.output_base_dir)
            data_dir = output_base / output_folder_name / "data"
            
            if not data_dir.exists():
                return "Data folder not found", "", "", ""
            
            txt_content = json_content = srt_content = vtt_content = ""
            
            # Find transcript files in data subfolder
            for file in data_dir.iterdir():
                if file.is_file():
                    if file.suffix == '.txt':
                        txt_content = file.read_text(encoding='utf-8')
                    elif file.suffix == '.json':
                        json_content = file.read_text(encoding='utf-8')
                    elif file.suffix == '.srt':
                        srt_content = file.read_text(encoding='utf-8')
                    elif file.suffix == '.vtt':
                        vtt_content = file.read_text(encoding='utf-8')
            
            return txt_content, json_content, srt_content, vtt_content
            
        except Exception as e:
            logger.error(f"Error loading output preview: {e}")
            return f"Error: {str(e)}", "", "", ""
    
    def get_output_folders(self) -> List[str]:
        """
        Get list of available output folders.
        
        Returns:
            List[str]: List of output folder names (without full path)
        """
        try:
            output_base = Path(self.file_manager.options.output_base_dir)
            if not output_base.exists():
                return []
            
            folders = [f.name for f in output_base.iterdir() if f.is_dir()]
            return sorted(folders, reverse=True)  # Most recent first
            
        except Exception as e:
            logger.error(f"Error listing output folders: {e}")
            return []
    
    def get_output_folders_with_latest_selected(self) -> Tuple[List[str], Optional[str]]:
        """
        Get list of output folders and return the latest one as selected.
        
        Returns:
            Tuple[List[str], Optional[str]]: (folders_list, latest_folder_name)
        """
        folders = self.get_output_folders()
        latest = folders[0] if folders else None
        return folders, latest
    
    def auto_create_zip(self, output_folder_name: str) -> Tuple[bool, str]:
        """
        Automatically create ZIP archive of output folder after transcription using multiprocessing.
        
        Args:
            output_folder_name: Name of output folder
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            output_base = Path(self.file_manager.options.output_base_dir)
            folder_path = output_base / output_folder_name
            
            if not folder_path.exists():
                return False, "Output folder not found"
            
            # Create queue for communication
            result_queue = Queue()
            
            # Start worker process
            process = Process(
                target=WhisperGUI.archive_worker,
                args=(str(folder_path), result_queue, output_folder_name)
            )
            
            process.start()
            logger.info(f"Archive worker process started (PID: {process.pid})")
            
            # Wait for result with timeout
            max_wait = 300  # 5 minutes timeout
            start_time = time.time()
            
            while process.is_alive() or not result_queue.empty():
                if not result_queue.empty():
                    status, archive_path, message = result_queue.get()
                    process.join(timeout=5)
                    
                    if status == "error":
                        logger.error(f"Archive creation failed: {message}")
                        return False, f"Archive creation failed: {message}"
                    elif status == "success":
                        logger.info(f"Archive created: {archive_path}")
                        return True, message
                
                # Check timeout
                if time.time() - start_time > max_wait:
                    logger.error("Archive creation timeout")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    return False, "Archive creation timeout (>5 minutes)"
                
                time.sleep(0.1)
            
            process.join(timeout=5)
            return False, "No result from archive worker"
            
        except Exception as e:
            logger.error(f"Error creating archive: {e}")
            return False, f"Archive creation failed: {str(e)}"
    
    @staticmethod
    def archive_worker(
        directory: str,
        result_queue: Queue,
        folder_name: str = None
    ):
        """
        Worker process for creating ZIP archives (runs in separate process to avoid blocking Gradio).
        Creates ZIP inside the folder as [name]/[name].zip
        """
        try:
            import zipfile
            from pathlib import Path
            from .config import SUPPORTED_FORMATS
            
            dir_path = Path(directory)
            
            if not dir_path.exists():
                result_queue.put(("error", "", f"Directory not found: {directory}"))
                return
            
            # Create ZIP inside the folder with the folder name
            if folder_name:
                archive_path = dir_path / f"{folder_name}.zip"
            else:
                archive_path = dir_path.parent / f"{dir_path.name}.zip"
            
            # Use zipfile with optimized settings
            with zipfile.ZipFile(str(archive_path), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zipf:
                for file in dir_path.rglob('*'):
                    if file.is_file():
                        rel_path = file.relative_to(dir_path)
                        
                        # Store media files, compress text files
                        compression = zipfile.ZIP_DEFLATED
                        if file.suffix.lower() in SUPPORTED_FORMATS:
                            compression = zipfile.ZIP_STORED
                            
                        zipf.write(file, rel_path, compress_type=compression)
            
            size_mb = archive_path.stat().st_size / (1024 * 1024)
            message = f"Archive created: {archive_path.name} ({size_mb:.2f} MB)"
            
            result_queue.put(("success", str(archive_path), message))
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[Archive Worker] Error: {e}", exc_info=True)
            result_queue.put(("error", "", f"Worker error: {str(e)}"))
    
    @staticmethod
    def transcription_worker(
        audio_path: str,
        model_size: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        use_vad: bool,
        word_timestamps: bool,
        use_batched: bool,
        batch_size: int,
        models_dir: str,
        result_queue: Queue,
        progress_queue: Queue
    ):
        """
        Worker process for transcription (runs in separate process to avoid CUDA threading issues).
        
        Single Responsibility: Only handles transcription, not file conversion.
        """
        try:
            # Send progress
            progress_queue.put(("log", "🔧 Initializing transcription service in worker process..."))
            
            # Initialize service in worker process
            config = ModelConfig(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                download_root=models_dir
            )
            service = TranscriptionService(config)
            
            progress_queue.put(("log", f"Loading model: {model_size} on {device}..."))
            success, init_msg = service.initialize_model()
            
            if not success:
                result_queue.put(("error", init_msg, None, None))
                return
            
            progress_queue.put(("log", init_msg))
            progress_queue.put(("log", "Transcribing..."))
            
            # Configure options
            trans_options = TranscriptionOptions(
                language=None if language == "auto" else language,
                beam_size=beam_size,
                vad_filter=use_vad,
                word_timestamps=word_timestamps
            )
            
            # Transcribe
            success, segments, metadata, trans_msg = service.transcribe(
                audio_path,
                trans_options,
                use_batched=use_batched,
                batch_size=batch_size,
                progress_callback=lambda msg: progress_queue.put(("log", f"  {msg}"))
            )
            
            if not success:
                result_queue.put(("error", trans_msg, None, None))
                return
            
            progress_queue.put(("log", trans_msg))
            
            # Return segments and metadata
            result_queue.put(("success", "Transcription complete", segments, metadata))
            
        except Exception as e:
            logger.error(f"[Worker] Error: {e}", exc_info=True)
            result_queue.put(("error", f"Worker error: {str(e)}", None, None))
    
    def transcribe_files(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        use_vad: bool,
        word_timestamps: bool,
        use_batched: bool,
        batch_size: int,
        output_formats: List[str],
        progress=gr.Progress()
    ):
        """
        Main transcription function using multiprocessing with progress updates.
        
        Yields progress updates to avoid blocking the UI.
        
        Returns:
            Tuple[str, str, str, str, str]: (progress_status, txt_output, json_output, srt_output, vtt_output)
        """
        current_progress = 0.0
        current_status = "Ready to start transcription..."
        
        def update_progress(value: float, desc: str = None):
            """Update progress bar and status."""
            nonlocal current_progress, current_status
            current_progress = value
            if desc:
                current_status = desc
                progress(value, desc=desc)
            else:
                progress(value)
            return current_status
        
        try:
            # Validate device selection
            update_progress(0.05, "Validating environment...")
            valid, msg = self.env_checker.validate_device_selection(device)
            if not valid:
                logger.error(f"Device validation failed: {msg}")
                yield f"❌ {msg}", "", "", "", ""
                return
            
            logger.info("Environment check passed")
            
            # Check if file selected
            if not self.current_file:
                logger.error("No file selected")
                yield "❌ No file selected from input folder", "", "", "", ""
                return
            
            file_path = Path(self.current_file)
            logger.info(f"Processing file: {file_path.name}")
            
            # Handle video to audio conversion (in main process, following SRP)
            audio_path = str(file_path)
            temp_audio_path = None
            
            if self.media_converter.is_video_file(str(file_path)):
                update_progress(0.1, "🎬 Extracting audio from video...")
                logger.info("Video file detected, extracting audio")
                
                # Create temp audio file
                temp_dir = Path(tempfile.gettempdir()) / "faster_whisper_gui"
                temp_dir.mkdir(exist_ok=True)
                temp_audio_path = str(temp_dir / f"{file_path.stem}_temp.wav")
                
                success, audio_path, conv_msg = self.media_converter.convert_video_to_audio(
                    str(file_path),
                    temp_audio_path,
                    audio_format="wav",
                    progress_callback=lambda msg: logger.info(f"FFmpeg: {msg}")
                )
                
                if not success:
                    logger.error(f"Video conversion failed: {conv_msg}")
                    yield f"❌ {conv_msg}", "", "", "", ""
                    return
            
            # Create queues for communication
            result_queue = Queue()
            progress_queue = Queue()
            
            # Start worker process (only handles transcription)
            update_progress(0.15, "⚙️ Starting transcription worker...")
            logger.info("Starting transcription worker process")
            
            process = Process(
                target=WhisperGUI.transcription_worker,
                args=(
                    audio_path,  # Pass audio path (converted or original)
                    model_size,
                    device,
                    compute_type,
                    language,
                    beam_size,
                    use_vad,
                    word_timestamps,
                    use_batched,
                    batch_size,
                    self.file_manager.options.models_dir,
                    result_queue,
                    progress_queue
                )
            )
            
            process.start()
            logger.info(f"Worker process started (PID: {process.pid})")
            
            # Monitor progress
            max_wait = 600  # 10 minutes timeout
            start_time = time.time()
            segments = None
            metadata = None
            last_yield_time = time.time()
            
            while process.is_alive() or not result_queue.empty() or not progress_queue.empty():
                # Check for progress messages
                has_new_messages = False
                while not progress_queue.empty():
                    msg_type, msg = progress_queue.get()
                    if msg_type == "log":
                        # Worker process log messages - only update progress status, not detailed log
                        has_new_messages = True
                        if "Loading model" in msg:
                            update_progress(0.2, "Loading model...")
                        elif "Transcribing" in msg and "complete" not in msg.lower():
                            update_progress(0.3, "Transcribing audio...")
                        elif "complete" in msg.lower():
                            update_progress(0.7, "Transcription complete")
                        elif "Initializing" in msg:
                            update_progress(0.18, "Initializing transcription...")
                    elif msg_type == "progress":
                        # Handle numeric progress updates
                        try:
                            prog_value = float(msg)
                            # Map 0-100 to 0.3-0.7 range (transcription phase)
                            update_progress(0.3 + (prog_value / 100) * 0.4, f"Transcribing... {prog_value:.0f}%")
                            has_new_messages = True
                        except:
                            pass
                
                # Check for result
                if not result_queue.empty():
                    status, message, segments, metadata = result_queue.get()
                    if status == "error":
                        logger.error(f"Transcription error: {message}")
                        yield f"❌ {message}", "", "", "", ""
                        process.join(timeout=5)
                        return
                    elif status == "success":
                        update_progress(0.75, "✅ Processing results...")
                        has_new_messages = True
                        break
                
                # Yield to UI only when there are new messages
                current_time = time.time()
                if has_new_messages:
                    yield current_status, "", "", "", ""
                    last_yield_time = current_time
                
                # Check timeout
                if time.time() - start_time > max_wait:
                    logger.error("Transcription timeout")
                    yield "❌ Transcription timeout (>10 minutes)", "", "", "", ""
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    return
                
                time.sleep(0.1)
            
            process.join(timeout=5)
            
            # Cleanup temp audio file (main process responsibility)
            if temp_audio_path and Path(temp_audio_path).exists():
                try:
                    Path(temp_audio_path).unlink()
                    logger.info(f"Cleaned up temp audio: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp audio: {e}")
            
            if segments is None or metadata is None:
                logger.error("No transcription result received")
                yield "❌ No transcription result received", "", "", "", ""
                return
            
            # Create output directory
            update_progress(0.78, "📂 Creating output directory...")
            success, output_dir, org_msg = self.file_manager.organize_output(str(file_path))
            if not success:
                logger.error(f"Failed to create output directory: {org_msg}")
                yield f"❌ {org_msg}", "", "", "", ""
                return
            
            self.current_output_dir = output_dir
            logger.info(f"Output directory: {Path(output_dir).name}")
            
            # Save outputs
            update_progress(0.80, "💾 Saving transcription files...")
            
            results = self.output_formatter.save_all_formats(
                segments,
                metadata,
                output_dir,
                "transcript",
                output_formats
            )
            
            txt_content = json_content = srt_content = vtt_content = ""
            
            for fmt, success in results.items():
                if success:
                    file_path_fmt = Path(output_dir) / f"transcript.{fmt}"
                    if file_path_fmt.exists():
                        content = file_path_fmt.read_text(encoding='utf-8')
                        if fmt == 'txt':
                            txt_content = content
                        elif fmt == 'json':
                            json_content = content
                        elif fmt == 'srt':
                            srt_content = content
                        elif fmt == 'vtt':
                            vtt_content = content
                    logger.info(f"Saved: {fmt.upper()}")
                else:
                    logger.warning(f"Failed to save: {fmt.upper()}")
            
            # Move original file to output (to data subfolder)
            if self.file_manager.options.move_input_to_output:
                update_progress(0.85, "📦 Moving original file...")
                success, new_path, move_msg = self.file_manager.move_input_to_output(
                    str(file_path),
                    output_dir
                )
                if success:
                    logger.info(move_msg)
            
            # Cleanup temp files
            self.file_manager.cleanup_temp_files()
            
            # Auto-create ZIP archive
            update_progress(0.90, "🗜️ Creating ZIP archive...")
            logger.info("Creating ZIP archive")
            
            zip_success, zip_msg = self.auto_create_zip(Path(output_dir).parent.name)
            if zip_success:
                logger.info(zip_msg)
            else:
                logger.warning(zip_msg)
            
            update_progress(1.0, "🎉 Transcription complete!")
            logger.info("Transcription complete")
            yield current_status, txt_content, json_content, srt_content, vtt_content
            
        except Exception as e:
            logger.exception("Transcription error")
            yield f"❌ Error: {str(e)}", "", "", "", ""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            gr.Blocks: Gradio interface
        """
        with gr.Blocks(title="Faster Whisper GUI") as interface:
            gr.Markdown("# Faster Whisper Transcription GUI")
            gr.Markdown("Upload audio/video files and transcribe them using faster-whisper with CUDA acceleration")
            
            with gr.Row():
                # Left column: Upload & Settings
                with gr.Column(scale=1):
                    gr.Markdown("## Upload Files")
                    file_upload = gr.File(
                        file_count="single",
                        label="Upload Audio/Video File",
                        file_types=[f for f in SUPPORTED_FORMATS]
                    )
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    
                    gr.Markdown("## Input Folder")
                    input_file_radio = gr.Radio(
                        label="Select File from Input Folder",
                        choices=[],
                        interactive=True
                    )
                    file_info = gr.Textbox(label="Selected File Info", interactive=False)
                    
                    gr.Markdown("## Transcription Settings")
                    
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=[m.value for m in ModelSize],
                            value=ModelSize.LARGE_V3.value,
                            label="Model Size",
                            info="Select Whisper model size. Larger models are more accurate but slower."
                        )
                        device_dropdown = gr.Dropdown(
                            choices=[DeviceType.CUDA.value, DeviceType.CPU.value],
                            value=DeviceType.CUDA.value,
                            label="Device",
                            info="Use CUDA for GPU acceleration or CPU for compatibility."
                        )
                    
                    with gr.Row():
                        compute_type_dropdown = gr.Dropdown(
                            choices=[c.value for c in ComputeType],
                            value=ComputeType.INT8_FLOAT16.value,
                            label="Compute Type",
                            info="Precision type for computation. int8_float16 balances speed and accuracy."
                        )
                        language_dropdown = gr.Dropdown(
                            choices=["auto"] + list(SUPPORTED_LANGUAGES.keys()),
                            value="auto",
                            label="Language",
                            info="Auto-detect language or specify manually for better accuracy."
                        )
                    
                    with gr.Row():
                        beam_size_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Beam Size",
                            info="Search width for decoding. Higher = more accurate but slower (1-10)."
                        )
                        batch_size_slider = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=8,
                            step=1,
                            label="Batch Size",
                            info="Number of samples processed in parallel when batched mode is enabled."
                        )
                    
                    with gr.Row():
                        use_vad_check = gr.Checkbox(
                            value=True,
                            label="VAD Filter",
                            info="Voice Activity Detection - removes silence/noise from transcription."
                        )
                        word_timestamps_check = gr.Checkbox(
                            value=False,
                            label="Word Timestamps",
                            info="Generate word-level timestamps (slower but more detailed)."
                        )
                        use_batched_check = gr.Checkbox(
                            value=True,
                            label="Batched Inference",
                            info="Process audio in batches for faster transcription on long files."
                        )
                    
                    output_formats_check = gr.CheckboxGroup(
                        choices=["txt", "json", "srt", "vtt"],
                        value=["txt", "json", "srt", "vtt"],
                        label="Output Formats",
                        info="Select which output formats to generate."
                    )
                    
                    transcribe_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")
                    
                   
                # Right column: Output & Preview
                with gr.Column(scale=1):
                    gr.Markdown("## 📊 Progress")
                    progress_display = gr.Textbox(
                        label="Current Status",
                        lines=1,
                        max_lines=1,
                        interactive=False,
                        show_label=False,
                        placeholder="Ready to start transcription..."
                    )
                    
                    gr.Markdown("## 📄 Output Preview")
                    output_folder_selector = gr.Radio(
                        label="Select Output Folder",
                        choices=[],
                        interactive=True
                    )
                    
                    with gr.Row():
                        download_file = gr.File(visible=False, scale=2, show_label=False)
                        move_back_btn = gr.Button("⬅️ Move Media Back to Input Folder", variant="secondary", size="sm", visible=False)

                    
                    move_status = gr.Textbox(label="Move Status", interactive=False, lines=1, visible=False)
                    
                    with gr.Tabs():
                        with gr.Tab("TXT"):
                            txt_output = gr.Textbox(
                                label="Plain Text Transcript",
                                lines=20,
                                max_lines=20,
                                interactive=False
                            )
                        
                        with gr.Tab("JSON"):
                            json_output = gr.Textbox(
                                label="JSON Transcript with Metadata",
                                lines=20,
                                max_lines=20,
                                interactive=False
                            )
                        
                        with gr.Tab("SRT"):
                            srt_output = gr.Textbox(
                                label="SRT Subtitle Format",
                                lines=20,
                                max_lines=20,
                                interactive=False
                            )
                        
                        with gr.Tab("VTT"):
                            vtt_output = gr.Textbox(
                                label="WebVTT Subtitle Format",
                                lines=20,
                                max_lines=20,
                                interactive=False
                            )
            
            # Event handlers
            file_upload.change(
                fn=self.upload_files,
                inputs=[file_upload],
                outputs=[upload_status, input_file_radio, file_info]
            )
            
            input_file_radio.change(
                fn=self.select_input_file,
                inputs=[input_file_radio],
                outputs=[file_info]
            )
            
            transcribe_btn.click(
                fn=self.transcribe_files,
                inputs=[
                    model_dropdown,
                    device_dropdown,
                    compute_type_dropdown,
                    language_dropdown,
                    beam_size_slider,
                    use_vad_check,
                    word_timestamps_check,
                    use_batched_check,
                    batch_size_slider,
                    output_formats_check
                ],
                outputs=[progress_display, txt_output, json_output, srt_output, vtt_output]
            ).then(
                fn=lambda: (
                    gr.Radio(choices=self.get_input_files_list()),
                    gr.Radio(choices=self.get_output_folders(), value=self.get_output_folders()[0] if self.get_output_folders() else None),
                    self.get_output_folders()[0] if self.get_output_folders() else None
                ),
                outputs=[input_file_radio, output_folder_selector, output_folder_selector]
            ).then(
                fn=self.download_output_folder,
                inputs=[output_folder_selector],
                outputs=[download_file, download_file, move_back_btn]
            )
            
            output_folder_selector.change(
                fn=self.load_output_preview,
                inputs=[output_folder_selector],
                outputs=[txt_output, json_output, srt_output, vtt_output]
            ).then(
                fn=self.download_output_folder,
                inputs=[output_folder_selector],
                outputs=[download_file, download_file, move_back_btn]
            )
            
            move_back_btn.click(
                fn=self.move_files_back,
                inputs=[output_folder_selector],
                outputs=[move_status]
            ).then(
                fn=lambda: (gr.Radio(choices=self.get_input_files_list()), gr.update(visible=True)),
                outputs=[input_file_radio, move_status]
            )
            
            # Initialize radio on load
            interface.load(
                fn=lambda: (
                    gr.Radio(choices=self.get_input_files_list()),
                    gr.Radio(choices=self.get_output_folders())
                ),
                outputs=[input_file_radio, output_folder_selector]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for gr.Blocks.launch()
        """
        interface = self.create_interface()
        # Set default theme if not provided
        if 'theme' not in kwargs:
            kwargs['theme'] = gr.themes.Soft()
        interface.launch(**kwargs)


def main():
    """Launch GUI application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    gui = WhisperGUI()
    
    # Validate environment
    valid, message = gui.validate_environment()
    if not valid:
        print(f"\n{message}\n")
        print("Please fix the environment issues before launching the GUI.")
        return
    
    print("✅ Environment validation passed")
    print("Launching Gradio interface...")
    
    gui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=1  # Force single-threaded to avoid CUDA threading issues
    )


if __name__ == "__main__":
    main()
