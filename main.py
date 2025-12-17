"""Main entry point for Faster Whisper GUI application."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.environment_checker import EnvironmentChecker
from src.gui import WhisperGUI


def setup_logging():
    """Configure application logging."""
    # Configure handlers with UTF-8 encoding for Windows compatibility
    import io
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('faster_whisper_gui.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )


def check_environment() -> bool:
    """
    Perform pre-flight environment checks.
    
    Returns:
        bool: True if environment is valid
    """
    print("\n" + "="*60)
    print("FASTER WHISPER GUI - Environment Check")
    print("="*60 + "\n")
    
    checker = EnvironmentChecker()
    
    # Print system report
    checker.print_system_report()
    
    # Validate environment
    # FFmpeg is required, CUDA is recommended but not required
    valid, message = checker.validate_environment(
        require_cuda=False,  # Don't require CUDA (user can select CPU)
        require_ffmpeg=True   # FFmpeg is required for video conversion
    )
    
    if not valid:
        print("\n" + "="*60)
        print("ENVIRONMENT VALIDATION FAILED")
        print("="*60)
        print(f"\n{message}\n")
        print("Please fix the issues above before running the application.")
        print("="*60 + "\n")
        return False
    
    print("\n" + "="*60)
    print("✅ ENVIRONMENT VALIDATION PASSED")
    print("="*60 + "\n")
    return True


def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Faster Whisper GUI")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    try:
        # Create and launch GUI
        print("Launching Gradio interface...")
        print("Access the GUI at: http://localhost:7860")
        print("Press Ctrl+C to stop the server\n")
        
        gui = WhisperGUI()
        gui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        logger.info("Application stopped by user")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
