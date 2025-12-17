"""Configuration settings and constants for faster-whisper GUI."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DeviceType(str, Enum):
    """Supported compute devices."""
    CUDA = "cuda"
    CPU = "cpu"


class ComputeType(str, Enum):
    """Supported compute precision types."""
    FLOAT16 = "float16"
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    FLOAT32 = "float32"


class AudioFormat(str, Enum):
    """Supported audio formats for conversion."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"


class ModelSize(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "turbo"
    DISTIL_LARGE_V3 = "distil-large-v3"


# File format constants
SUPPORTED_AUDIO_FORMATS = [
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"
]

SUPPORTED_VIDEO_FORMATS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"
]

SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS


# Default configuration values
DEFAULT_DEVICE = DeviceType.CUDA
DEFAULT_COMPUTE_TYPE_CUDA = ComputeType.FLOAT16
DEFAULT_COMPUTE_TYPE_CPU = ComputeType.INT8
DEFAULT_MODEL_SIZE = ModelSize.LARGE_V3
DEFAULT_BEAM_SIZE = 5
DEFAULT_AUDIO_FORMAT = AudioFormat.WAV


@dataclass
class ModelConfig:
    """Configuration for Whisper model initialization."""
    model_size: str = DEFAULT_MODEL_SIZE.value
    device: str = DEFAULT_DEVICE.value
    compute_type: str = DEFAULT_COMPUTE_TYPE_CUDA.value
    download_root: Optional[str] = "models"
    local_files_only: bool = False
    
    def get_compute_type_for_device(self) -> str:
        """Get appropriate compute type based on device."""
        if self.device == DeviceType.CUDA.value:
            return DEFAULT_COMPUTE_TYPE_CUDA.value
        return DEFAULT_COMPUTE_TYPE_CPU.value


@dataclass
class TranscriptionOptions:
    """Options for audio transcription."""
    language: Optional[str] = None
    beam_size: int = DEFAULT_BEAM_SIZE
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: List[int] = field(default_factory=lambda: [-1])
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'\"¿([{-"
    append_punctuations: str = "\"'\".。,，!！?？:：\")]}、"
    vad_filter: bool = True
    vad_parameters: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for faster-whisper API."""
        return {
            "language": self.language,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self.initial_prompt,
            "prefix": self.prefix,
            "suppress_blank": self.suppress_blank,
            "suppress_tokens": self.suppress_tokens,
            "without_timestamps": self.without_timestamps,
            "max_initial_timestamp": self.max_initial_timestamp,
            "word_timestamps": self.word_timestamps,
            "prepend_punctuations": self.prepend_punctuations,
            "append_punctuations": self.append_punctuations,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
        }


@dataclass
class ConversionOptions:
    """Options for media conversion."""
    audio_format: str = DEFAULT_AUDIO_FORMAT.value
    audio_bitrate: str = "192k"
    sample_rate: int = 16000
    channels: int = 1  # mono for transcription
    keep_extracted_audio: bool = False


@dataclass
class FileOrganizationOptions:
    """Options for file organization."""
    input_dir: str = "input"
    output_base_dir: str = "output"
    temp_dir: str = "temp"
    create_subfolder: bool = True
    move_input_to_output: bool = True
    subfolder_name_format: str = "{filename}"  # {filename}, {timestamp}, etc.
    models_dir: str = "models"  # Directory for storing downloaded models


@dataclass
class OutputFormats:
    """Output format settings."""
    generate_txt: bool = True
    generate_json: bool = True
    generate_srt: bool = True
    txt_filename: str = "transcript.txt"
    json_filename: str = "transcript.json"
    srt_filename: str = "transcript.srt"


# Language codes supported by Whisper
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}


def get_default_config() -> dict:
    """Get default configuration for the application."""
    return {
        "model": ModelConfig(),
        "transcription": TranscriptionOptions(),
        "conversion": ConversionOptions(),
        "organization": FileOrganizationOptions(),
        "output_formats": OutputFormats(),
    }
