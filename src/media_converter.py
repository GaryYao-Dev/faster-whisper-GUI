"""Media conversion utilities for video-to-audio extraction using FFmpeg."""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import ffmpeg

from .config import AudioFormat, ConversionOptions, SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)


class MediaConverter:
    """Handles video-to-audio conversion using FFmpeg."""
    
    def __init__(self, options: Optional[ConversionOptions] = None):
        """
        Initialize MediaConverter.
        
        Args:
            options: Conversion options. Uses defaults if None.
        """
        self.options = options or ConversionOptions()
    
    def is_video_file(self, file_path: str) -> bool:
        """
        Check if file is a video format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if video file
        """
        path = Path(file_path)
        return path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
    
    def get_audio_codec_info(self, file_path: str) -> dict:
        """
        Get audio stream information from media file.
        
        Args:
            file_path: Path to media file
            
        Returns:
            dict: Audio stream information
        """
        try:
            probe = ffmpeg.probe(file_path)
            audio_streams = [
                stream for stream in probe['streams']
                if stream['codec_type'] == 'audio'
            ]
            
            if not audio_streams:
                return {"has_audio": False}
            
            audio_info = audio_streams[0]
            return {
                "has_audio": True,
                "codec": audio_info.get('codec_name', 'unknown'),
                "sample_rate": audio_info.get('sample_rate', 'unknown'),
                "channels": audio_info.get('channels', 'unknown'),
                "bitrate": audio_info.get('bit_rate', 'unknown'),
            }
        except Exception as e:
            logger.error(f"Error probing file {file_path}: {e}")
            return {"has_audio": False, "error": str(e)}
    
    def convert_video_to_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        audio_format: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, str, str]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file (optional)
            audio_format: Output audio format (optional, uses config default)
            progress_callback: Optional callback for progress messages
            
        Returns:
            Tuple[bool, str, str]: (success, output_path, message)
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return False, "", f"Video file not found: {video_path}"
            
            # Determine output path
            if output_path is None:
                fmt = audio_format or self.options.audio_format
                output_path = video_path.parent / f"{video_path.stem}.{fmt}"
            else:
                output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback(f"Converting {video_path.name} to audio...")
            logger.info(f"Converting {video_path.name} to audio...")
            
            # Build ffmpeg command
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le' if audio_format == 'wav' else 'libmp3lame',
                ar=self.options.sample_rate,
                ac=self.options.channels,
                audio_bitrate=self.options.audio_bitrate,
                loglevel='warning'  # Changed from 'error' to see progress
            )
            
            if progress_callback:
                progress_callback(f"Running FFmpeg conversion...")
            
            # Run conversion
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                message = f"✅ Audio extracted: {output_path.name} ({size_mb:.2f} MB)"
                logger.info(message)
                if progress_callback:
                    progress_callback(message)
                return True, str(output_path), message
            else:
                message = "Conversion completed but output file not found"
                logger.error(message)
                return False, "", message
                
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            message = f"FFmpeg error: {error_msg}"
            logger.error(message)
            return False, "", message
        except Exception as e:
            message = f"Error converting video to audio: {str(e)}"
            logger.error(message)
            return False, "", message
    
    def batch_convert(
        self,
        file_list: list,
        output_dir: Optional[str] = None,
        audio_format: Optional[str] = None
    ) -> list:
        """
        Convert multiple video files to audio.
        
        Args:
            file_list: List of video file paths
            output_dir: Directory for output files (optional)
            audio_format: Output audio format (optional)
            
        Returns:
            list: List of tuples (success, input_path, output_path, message)
        """
        results = []
        
        for video_path in file_list:
            if not self.is_video_file(video_path):
                results.append((
                    False,
                    video_path,
                    "",
                    f"Not a video file: {Path(video_path).name}"
                ))
                continue
            
            # Determine output path
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                fmt = audio_format or self.options.audio_format
                output_path = output_dir_path / f"{Path(video_path).stem}.{fmt}"
            else:
                output_path = None
            
            success, out_path, message = self.convert_video_to_audio(
                video_path,
                str(output_path) if output_path else None,
                audio_format
            )
            
            results.append((success, video_path, out_path, message))
        
        return results
    
    def needs_conversion(self, file_path: str) -> bool:
        """
        Check if file needs conversion (is video).
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if conversion needed
        """
        return self.is_video_file(file_path)
    
    def prepare_audio_for_transcription(
        self,
        file_path: str,
        temp_dir: str
    ) -> Tuple[bool, str, str]:
        """
        Prepare audio file for transcription (convert if video).
        
        Args:
            file_path: Path to input file
            temp_dir: Temporary directory for converted files
            
        Returns:
            Tuple[bool, str, str]: (success, audio_path, message)
        """
        path = Path(file_path)
        
        # If already audio, return as-is
        if not self.is_video_file(file_path):
            return True, file_path, f"Audio file ready: {path.name}"
        
        # Convert video to audio
        temp_dir_path = Path(temp_dir)
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_path = temp_dir_path / f"{path.stem}.{self.options.audio_format}"
        
        success, audio_path, message = self.convert_video_to_audio(
            file_path,
            str(output_path)
        )
        
        return success, audio_path, message


def main():
    """Test media converter functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    converter = MediaConverter()
    
    # Example usage
    print("MediaConverter initialized")
    print(f"Default audio format: {converter.options.audio_format}")
    print(f"Sample rate: {converter.options.sample_rate}")
    print(f"Channels: {converter.options.channels}")


if __name__ == "__main__":
    main()
