"""Transcription service wrapper for faster-whisper."""

import logging
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
from faster_whisper import WhisperModel, BatchedInferencePipeline

from .config import ModelConfig, TranscriptionOptions, ModelSize

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Wrapper for faster-whisper transcription functionality."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize TranscriptionService.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.model: Optional[WhisperModel] = None
        self.batched_model: Optional[BatchedInferencePipeline] = None
        self._is_initialized = False
    
    def initialize_model(self) -> Tuple[bool, str]:
        """
        Initialize the Whisper model.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            logger.info(f"Loading model: {self.config.model_size}")
            logger.info(f"Device: {self.config.device}")
            logger.info(f"Compute type: {self.config.compute_type}")
            
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=self.config.download_root,
                local_files_only=self.config.local_files_only,
            )
            
            self._is_initialized = True
            message = f"Model '{self.config.model_size}' loaded successfully on {self.config.device}"
            logger.info(message)
            return True, message
            
        except Exception as e:
            message = f"Error loading model: {str(e)}"
            logger.error(message)
            return False, message
    
    def initialize_batched_model(self) -> Tuple[bool, str]:
        """
        Initialize batched inference pipeline for faster processing.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self._is_initialized or self.model is None:
                success, msg = self.initialize_model()
                if not success:
                    return False, msg
            
            logger.info("Initializing batched inference pipeline...")
            self.batched_model = BatchedInferencePipeline(model=self.model)
            
            message = "Batched inference pipeline initialized"
            logger.info(message)
            return True, message
            
        except Exception as e:
            message = f"Error initializing batched pipeline: {str(e)}"
            logger.error(message)
            return False, message
    
    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None,
        use_batched: bool = False,
        batch_size: int = 16,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, List[Any], Dict[str, Any], str]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            options: Transcription options. Uses defaults if None.
            use_batched: Use batched inference for faster processing
            batch_size: Batch size for batched inference
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple[bool, segments, metadata, message]
        """
        try:
            if not Path(audio_path).exists():
                return False, [], {}, f"Audio file not found: {audio_path}"
            
            # Initialize model if not already done
            if not self._is_initialized:
                success, msg = self.initialize_model()
                if not success:
                    return False, [], {}, msg
            
            # Use batched model if requested
            if use_batched:
                if self.batched_model is None:
                    success, msg = self.initialize_batched_model()
                    if not success:
                        return False, [], {}, msg
                transcription_model = self.batched_model
            else:
                transcription_model = self.model
            
            # Prepare transcription options
            trans_options = options or TranscriptionOptions()
            options_dict = trans_options.to_dict()
            
            # Add batch_size for batched inference
            if use_batched:
                options_dict['batch_size'] = batch_size
            
            logger.info(f"Transcribing: {Path(audio_path).name}")
            logger.info(f"Transcription parameters: audio_path={audio_path}")
            logger.info(f"  use_batched={use_batched}, batch_size={batch_size}")
            logger.info(f"  options_dict={options_dict}")
            logger.info(f"  model_type={'batched' if use_batched else 'standard'}")
            
            # Force CUDA context in current thread if using CUDA
            if self.config.device == "cuda" and torch.cuda.is_available():
                logger.info(f"Setting CUDA device in current thread: cuda:0")
                torch.cuda.set_device(0)
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            
            if progress_callback:
                logger.info("Calling progress_callback with 'Initializing transcription pipeline...'")
                progress_callback("Initializing transcription pipeline...")
                logger.info("progress_callback returned")
            
            logger.info("About to call transcription_model.transcribe()...")
            # Perform transcription - this call returns immediately with a generator
            segments_generator, info = transcription_model.transcribe(
                audio_path,
                **options_dict
            )
            logger.info("transcription_model.transcribe() returned generator")
            
            logger.info("Transcription pipeline ready, iterating segments (this may take a while for first segment)...")
            if progress_callback:
                progress_callback("Processing audio (this may take a moment)...")
            
            # Convert generator to list - THIS is where the actual work happens
            segments = []
            for i, segment in enumerate(segments_generator):
                segments.append(segment)
                # Log first segment separately to show when actual processing starts
                if i == 0:
                    logger.info("First segment received, transcription in progress...")
                    if progress_callback:
                        progress_callback("Transcription started, processing segments...")
                if progress_callback and i > 0 and i % 10 == 0:
                    progress_callback(f"Processing segment {i + 1}...")
            
            # Prepare metadata
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') else None,
                "model_size": self.config.model_size,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
                "transcription_options": {
                    "beam_size": trans_options.beam_size,
                    "vad_filter": trans_options.vad_filter,
                    "word_timestamps": trans_options.word_timestamps,
                }
            }
            
            segment_count = len(segments)
            duration = info.duration
            message = (
                f"Transcription complete: {segment_count} segments, "
                f"{duration:.2f}s duration, "
                f"language: {info.language} ({info.language_probability:.2%})"
            )
            logger.info(message)
            
            if progress_callback:
                progress_callback(message)
            
            return True, segments, metadata, message
            
        except Exception as e:
            message = f"Error during transcription: {str(e)}"
            logger.error(message)
            return False, [], {}, message
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model sizes.
        
        Returns:
            List[str]: Available model names
        """
        return [model.value for model in ModelSize]
    
    def update_config(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ) -> bool:
        """
        Update model configuration.
        
        Args:
            model_size: New model size
            device: New device
            compute_type: New compute type
            
        Returns:
            bool: True if model needs reinitialization
        """
        needs_reinit = False
        
        if model_size and model_size != self.config.model_size:
            self.config.model_size = model_size
            needs_reinit = True
        
        if device and device != self.config.device:
            self.config.device = device
            needs_reinit = True
        
        if compute_type and compute_type != self.config.compute_type:
            self.config.compute_type = compute_type
            needs_reinit = True
        
        if needs_reinit:
            self._is_initialized = False
            self.model = None
            self.batched_model = None
            logger.info("Model configuration updated - reinitialization required")
        
        return needs_reinit
    
    def unload_model(self):
        """Unload model to free memory."""
        try:
            self.model = None
            self.batched_model = None
            self._is_initialized = False
            logger.info("Model unloaded")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information.
        
        Returns:
            Dict: Model information
        """
        return {
            "model_size": self.config.model_size,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "is_initialized": self._is_initialized,
            "has_batched_pipeline": self.batched_model is not None,
        }


def main():
    """Test transcription service."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create service with default config
    config = ModelConfig(
        model_size="tiny",  # Use tiny for quick testing
        device="cpu",
        compute_type="int8"
    )
    
    service = TranscriptionService(config)
    
    print("TranscriptionService Test")
    print(f"Model: {service.config.model_size}")
    print(f"Device: {service.config.device}")
    print(f"Available models: {service.get_available_models()}")
    
    # Test initialization
    success, message = service.initialize_model()
    print(f"\nInitialization: {'✅' if success else '❌'} {message}")


if __name__ == "__main__":
    main()
