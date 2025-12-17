"""Environment validation for CUDA and FFmpeg requirements."""

import subprocess
import sys
import logging
from typing import Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentChecker:
    """Validates system environment for faster-whisper transcription."""
    
    def __init__(self):
        self.cuda_available = False
        self.cuda_version = None
        self.ffmpeg_available = False
        self.ffmpeg_version = None
    
    def check_cuda(self) -> Tuple[bool, str]:
        """
        Check if CUDA is available.
        
        Returns:
            Tuple[bool, str]: (is_available, version_or_error_message)
        """
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.cuda_version = torch.version.cuda
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                message = f"CUDA {self.cuda_version} - {device_name} (x{device_count})"
                logger.info(f"CUDA available: {message}")
                return True, message
            else:
                # Check if CUDA toolkit is installed but PyTorch doesn't have CUDA support
                cuda_version = torch.version.cuda
                if cuda_version is None:
                    message = "PyTorch installed without CUDA support. Install torch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
                else:
                    message = f"CUDA toolkit found but no GPU detected. Check: 1) GPU drivers installed? 2) GPU enabled in BIOS? 3) Run 'nvidia-smi' to verify"
                logger.warning(message)
                return False, message
        except ImportError:
            message = "PyTorch not installed - cannot detect CUDA"
            logger.warning(message)
            return False, message
        except Exception as e:
            message = f"Error checking CUDA: {str(e)}"
            logger.error(message)
            return False, message
    
    def check_ffmpeg(self) -> Tuple[bool, str]:
        """
        Check if FFmpeg is installed and accessible.
        
        Returns:
            Tuple[bool, str]: (is_available, version_or_error_message)
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split('\n')[0]
                self.ffmpeg_version = version_line.split()[2] if len(version_line.split()) > 2 else "unknown"
                self.ffmpeg_available = True
                message = f"FFmpeg {self.ffmpeg_version}"
                logger.info(f"FFmpeg available: {message}")
                return True, message
            else:
                message = "FFmpeg found but returned error"
                logger.error(message)
                return False, message
        except FileNotFoundError:
            message = "FFmpeg not found in PATH"
            logger.error(message)
            return False, message
        except subprocess.TimeoutExpired:
            message = "FFmpeg check timeout"
            logger.error(message)
            return False, message
        except Exception as e:
            message = f"Error checking FFmpeg: {str(e)}"
            logger.error(message)
            return False, message
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if required Python packages are installed.
        
        Returns:
            Dict[str, bool]: Package availability status
        """
        dependencies = {}
        packages = [
            "faster_whisper",
            "gradio",
            "ffmpeg",
            "torch"
        ]
        
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
                dependencies[package] = True
                logger.info(f"Package '{package}' is installed")
            except ImportError:
                dependencies[package] = False
                logger.warning(f"Package '{package}' is NOT installed")
        
        return dependencies
    
    def get_system_info(self) -> Dict[str, any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict: System information including CUDA, FFmpeg, Python version
        """
        cuda_ok, cuda_msg = self.check_cuda()
        ffmpeg_ok, ffmpeg_msg = self.check_ffmpeg()
        deps = self.check_dependencies()
        
        return {
            "python_version": sys.version.split()[0],
            "cuda_available": cuda_ok,
            "cuda_info": cuda_msg,
            "ffmpeg_available": ffmpeg_ok,
            "ffmpeg_info": ffmpeg_msg,
            "dependencies": deps,
            "recommended_device": self.get_recommended_device()
        }
    
    def get_recommended_device(self) -> str:
        """
        Get recommended device based on CUDA availability.
        
        Returns:
            str: "cuda" or "cpu"
        """
        if self.cuda_available:
            return "cuda"
        return "cpu"
    
    def validate_environment(self, require_cuda: bool = True, require_ffmpeg: bool = True) -> Tuple[bool, str]:
        """
        Validate complete environment with strict requirements.
        
        Args:
            require_cuda: If True, exit if CUDA not available (unless CPU explicitly selected)
            require_ffmpeg: If True, exit if FFmpeg not available
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        issues = []
        
        # Check FFmpeg (required)
        ffmpeg_ok, ffmpeg_msg = self.check_ffmpeg()
        if require_ffmpeg and not ffmpeg_ok:
            issues.append(
                f"❌ FFmpeg is required but not found.\n"
                f"   Please install FFmpeg and add it to your PATH.\n"
                f"   Download: https://ffmpeg.org/download.html"
            )
        
        # Check CUDA (recommended, but not strictly required if CPU mode is used)
        cuda_ok, cuda_msg = self.check_cuda()
        if require_cuda and not cuda_ok:
            issues.append(
                f"⚠️  CUDA not available: {cuda_msg}\n"
                f"   For GPU acceleration, install CUDA drivers.\n"
                f"   You can still use CPU mode by explicitly selecting it."
            )
        
        if issues:
            return False, "\n\n".join(issues)
        
        return True, "Environment validation passed"
    
    def validate_device_selection(self, device: str) -> Tuple[bool, str]:
        """
        Validate device selection before transcription.
        
        Args:
            device: "cuda" or "cpu"
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if device.lower() == "cuda":
            cuda_ok, cuda_msg = self.check_cuda()
            if not cuda_ok:
                return False, (
                    f"Cannot use CUDA: {cuda_msg}\n"
                    f"Please select CPU mode or install CUDA drivers."
                )
        
        return True, f"Device '{device}' is available"
    
    def print_system_report(self):
        """Print a formatted system report to console."""
        info = self.get_system_info()
        
        print("\n" + "="*60)
        print("SYSTEM ENVIRONMENT REPORT")
        print("="*60)
        print(f"Python Version:     {info['python_version']}")
        print(f"CUDA Available:     {'✅' if info['cuda_available'] else '❌'} {info['cuda_info']}")
        print(f"FFmpeg Available:   {'✅' if info['ffmpeg_available'] else '❌'} {info['ffmpeg_info']}")
        print(f"Recommended Device: {info['recommended_device'].upper()}")
        print("\nDependencies:")
        for pkg, installed in info['dependencies'].items():
            status = "✅" if installed else "❌"
            print(f"  {status} {pkg}")
        print("="*60 + "\n")


def main():
    """Run environment checks as standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    checker = EnvironmentChecker()
    checker.print_system_report()
    
    # Validate with strict requirements
    valid, message = checker.validate_environment(require_cuda=False, require_ffmpeg=True)
    if not valid:
        print(f"\n{message}\n")
        sys.exit(1)
    else:
        print(f"\n✅ {message}\n")


if __name__ == "__main__":
    main()
