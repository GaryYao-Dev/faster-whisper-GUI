"""Output formatting for transcription results in TXT, JSON, and SRT formats."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Formats transcription segments into various output formats."""
    
    @staticmethod
    def format_timestamp(seconds: float, srt_format: bool = False) -> str:
        """
        Format timestamp for display.
        
        Args:
            seconds: Time in seconds
            srt_format: If True, use SRT format (HH:MM:SS,mmm), else simple format
            
        Returns:
            str: Formatted timestamp
        """
        if srt_format:
            # SRT format: HH:MM:SS,mmm
            td = timedelta(seconds=seconds)
            hours = int(td.total_seconds() // 3600)
            minutes = int((td.total_seconds() % 3600) // 60)
            secs = int(td.total_seconds() % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        else:
            # Simple format: MM:SS or HH:MM:SS
            td = timedelta(seconds=seconds)
            hours = int(td.total_seconds() // 3600)
            minutes = int((td.total_seconds() % 3600) // 60)
            secs = td.total_seconds() % 60
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
            else:
                return f"{minutes:02d}:{secs:05.2f}"
    
    def to_txt(
        self,
        segments: List[Any],
        output_path: str,
        include_timestamps: bool = True
    ) -> bool:
        """
        Save transcription as plain text file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output TXT file
            include_timestamps: Whether to include timestamps
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    if include_timestamps:
                        start = self.format_timestamp(segment.start)
                        end = self.format_timestamp(segment.end)
                        f.write(f"[{start} -> {end}] {segment.text.strip()}\n")
                    else:
                        f.write(f"{segment.text.strip()}\n")
            
            logger.info(f"TXT file saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving TXT file: {e}")
            return False
    
    def to_json(
        self,
        segments: List[Any],
        metadata: Optional[Dict[str, Any]],
        output_path: str,
        include_words: bool = False
    ) -> bool:
        """
        Save transcription as JSON file with metadata.
        
        Args:
            segments: List of transcription segments
            metadata: Transcription metadata (language, duration, etc.)
            output_path: Path to output JSON file
            include_words: Whether to include word-level timestamps
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build JSON structure
            data = {
                "metadata": metadata or {},
                "segments": []
            }
            
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                
                # Add word-level timestamps if available
                if include_words and hasattr(segment, 'words') and segment.words:
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]
                
                data["segments"].append(segment_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON file saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")
            return False
    
    def to_srt(
        self,
        segments: List[Any],
        output_path: str
    ) -> bool:
        """
        Save transcription as SRT subtitle file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output SRT file
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, start=1):
                    # SRT format:
                    # 1
                    # 00:00:00,000 --> 00:00:02,500
                    # Text content
                    # (blank line)
                    
                    start_time = self.format_timestamp(segment.start, srt_format=True)
                    end_time = self.format_timestamp(segment.end, srt_format=True)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text.strip()}\n")
                    f.write("\n")
            
            logger.info(f"SRT file saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving SRT file: {e}")
            return False
    
    def to_vtt(
        self,
        segments: List[Any],
        output_path: str
    ) -> bool:
        """
        Save transcription as WebVTT subtitle file.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output VTT file
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for segment in segments:
                    # VTT format (similar to SRT but with dots instead of commas)
                    start_time = self.format_timestamp(segment.start, srt_format=True).replace(',', '.')
                    end_time = self.format_timestamp(segment.end, srt_format=True).replace(',', '.')
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text.strip()}\n")
                    f.write("\n")
            
            logger.info(f"VTT file saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving VTT file: {e}")
            return False
    
    def save_all_formats(
        self,
        segments: List[Any],
        metadata: Optional[Dict[str, Any]],
        output_dir: str,
        base_filename: str = "transcript",
        formats: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Save transcription in all specified formats.
        
        Args:
            segments: List of transcription segments
            metadata: Transcription metadata
            output_dir: Output directory
            base_filename: Base filename without extension
            formats: List of formats to generate (txt, json, srt, vtt). If None, generates all.
            
        Returns:
            Dict[str, bool]: Success status for each format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ["txt", "json", "srt"]
        
        results = {}
        
        if "txt" in formats:
            txt_path = output_dir / f"{base_filename}.txt"
            results["txt"] = self.to_txt(segments, str(txt_path))
        
        if "json" in formats:
            json_path = output_dir / f"{base_filename}.json"
            results["json"] = self.to_json(segments, metadata, str(json_path))
        
        if "srt" in formats:
            srt_path = output_dir / f"{base_filename}.srt"
            results["srt"] = self.to_srt(segments, str(srt_path))
        
        if "vtt" in formats:
            vtt_path = output_dir / f"{base_filename}.vtt"
            results["vtt"] = self.to_vtt(segments, str(vtt_path))
        
        return results


def main():
    """Test output formatter functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mock segment class for testing
    class MockSegment:
        def __init__(self, id, start, end, text):
            self.id = id
            self.start = start
            self.end = end
            self.text = text
            self.avg_logprob = -0.5
            self.no_speech_prob = 0.1
    
    # Create test segments
    segments = [
        MockSegment(0, 0.0, 2.5, "Hello, this is a test."),
        MockSegment(1, 2.5, 5.0, "Testing the output formatter."),
        MockSegment(2, 5.0, 8.0, "This should create multiple format files."),
    ]
    
    metadata = {
        "language": "en",
        "duration": 8.0,
        "model": "large-v3"
    }
    
    formatter = OutputFormatter()
    
    # Test all formats
    results = formatter.save_all_formats(
        segments,
        metadata,
        "test_output",
        "test_transcript",
        ["txt", "json", "srt", "vtt"]
    )
    
    print("Output Formatter Test Results:")
    for fmt, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {fmt.upper()}")


if __name__ == "__main__":
    main()
