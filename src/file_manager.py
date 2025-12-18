"""File management utilities for organizing input/output files."""

import shutil
import logging
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List
import datetime

from .config import SUPPORTED_FORMATS, FileOrganizationOptions

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file organization, moving, and cleanup operations."""
    
    def __init__(self, options: Optional[FileOrganizationOptions] = None):
        """
        Initialize FileManager.
        
        Args:
            options: File organization options. Uses defaults if None.
        """
        self.options = options or FileOrganizationOptions()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary base directories."""
        Path(self.options.input_dir).mkdir(parents=True, exist_ok=True)
        Path(self.options.output_base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.options.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.options.models_dir).mkdir(parents=True, exist_ok=True)
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if file is a supported audio/video format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return False, f"File not found: {file_path}"
            
            if not path.is_file():
                return False, f"Not a file: {file_path}"
            
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                return False, f"Unsupported format: {path.suffix}"
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb < 0.001:
                return False, f"File too small: {size_mb:.3f} MB"
            
            return True, f"Valid file: {path.name} ({size_mb:.2f} MB)"
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def get_output_subfolder_path(self, input_file_path: str) -> Path:
        """
        Get the output subfolder path for an input file.
        
        Args:
            input_file_path: Path to input file
            
        Returns:
            Path: Output subfolder path
        """
        input_path = Path(input_file_path)
        
        # Format subfolder name
        if self.options.subfolder_name_format == "{filename}":
            subfolder_name = input_path.stem
        elif self.options.subfolder_name_format == "{timestamp}":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"{input_path.stem}_{timestamp}"
        else:
            subfolder_name = input_path.stem
        
        return Path(self.options.output_base_dir) / subfolder_name
    
    def organize_output(
        self,
        input_file_path: str,
        create_subfolder: Optional[bool] = None
    ) -> Tuple[bool, str, str]:
        """
        Create output directory structure for a file.
        Creates: [name]/data/ subfolder for transcription files.
        
        Args:
            input_file_path: Path to input file
            create_subfolder: Override default subfolder creation setting
            
        Returns:
            Tuple[bool, str, str]: (success, data_dir_path, message)
        """
        try:
            if create_subfolder is None:
                create_subfolder = self.options.create_subfolder
            
            if create_subfolder:
                base_output_dir = self.get_output_subfolder_path(input_file_path)
                # Create data subfolder
                output_dir = base_output_dir / "data"
            else:
                output_dir = Path(self.options.output_base_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            message = f"Output directory ready: {output_dir}"
            logger.info(message)
            return True, str(output_dir), message
            
        except Exception as e:
            message = f"Error creating output directory: {str(e)}"
            logger.error(message)
            return False, "", message
    
    def move_input_to_output(
        self,
        input_file_path: str,
        output_subfolder: str
    ) -> Tuple[bool, str, str]:
        """
        Move input file to output subfolder after processing.
        
        Args:
            input_file_path: Path to input file
            output_subfolder: Destination subfolder path
            
        Returns:
            Tuple[bool, str, str]: (success, new_path, message)
        """
        try:
            input_path = Path(input_file_path)
            output_dir = Path(output_subfolder)
            
            if not input_path.exists():
                return False, "", f"Input file not found: {input_file_path}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / input_path.name
            
            # Handle duplicate names
            counter = 1
            while destination.exists():
                destination = output_dir / f"{input_path.stem}_{counter}{input_path.suffix}"
                counter += 1
            
            shutil.move(str(input_path), str(destination))
            
            message = f"Moved: {input_path.name} -> {output_dir.name}/"
            logger.info(message)
            return True, str(destination), message
            
        except Exception as e:
            message = f"Error moving file: {str(e)}"
            logger.error(message)
            return False, "", message
    
    def move_back_to_input(
        self,
        file_path: str,
        input_dir: str
    ) -> Tuple[bool, str, str]:
        """
        Restore file from output back to input directory.
        
        Args:
            file_path: Current path to file in output
            input_dir: Destination input directory
            
        Returns:
            Tuple[bool, str, str]: (success, new_path, message)
        """
        try:
            file_path = Path(file_path)
            input_dir = Path(input_dir)
            
            if not file_path.exists():
                return False, "", f"File not found: {file_path}"
            
            input_dir.mkdir(parents=True, exist_ok=True)
            destination = input_dir / file_path.name
            
            # Handle duplicate names
            counter = 1
            while destination.exists():
                destination = input_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
            
            shutil.move(str(file_path), str(destination))
            
            message = f"Restored: {file_path.name} -> {input_dir}/"
            logger.info(message)
            return True, str(destination), message
            
        except Exception as e:
            message = f"Error restoring file: {str(e)}"
            logger.error(message)
            return False, "", message
    
    def copy_file(
        self,
        source_path: str,
        destination_path: str
    ) -> Tuple[bool, str]:
        """
        Copy file to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            source = Path(source_path)
            destination = Path(destination_path)
            
            if not source.exists():
                return False, f"Source file not found: {source_path}"
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
            
            message = f"Copied: {source.name} -> {destination}"
            logger.info(message)
            return True, message
            
        except Exception as e:
            message = f"Error copying file: {str(e)}"
            logger.error(message)
            return False, message
    
    def copy_to_input_folder(self, source_path: str) -> Tuple[bool, str, str]:
        """
        Copy uploaded file to input folder.
        
        Args:
            source_path: Path to uploaded file
            
        Returns:
            Tuple[bool, str, str]: (success, new_path, message)
        """
        try:
            source = Path(source_path)
            input_dir = Path(self.options.input_dir)
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # Create destination path
            destination = input_dir / source.name
            
            # Handle duplicates
            counter = 1
            while destination.exists():
                destination = input_dir / f"{source.stem}_{counter}{source.suffix}"
                counter += 1
            
            shutil.copy2(str(source), str(destination))
            
            message = f"File copied to input: {destination.name}"
            logger.info(message)
            return True, str(destination), message
            
        except Exception as e:
            message = f"Error copying to input folder: {str(e)}"
            logger.error(message)
            return False, "", message
    
    def cleanup_temp_files(self, temp_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Clean up temporary files.
        
        Args:
            temp_dir: Temporary directory to clean (uses default if None)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            temp_path = Path(temp_dir) if temp_dir else Path(self.options.temp_dir)
            
            if not temp_path.exists():
                return True, "Temp directory does not exist (already clean)"
            
            # Remove all files in temp directory
            file_count = 0
            for item in temp_path.iterdir():
                if item.is_file():
                    item.unlink()
                    file_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    file_count += 1
            
            message = f"Cleaned up {file_count} items from temp directory"
            logger.info(message)
            return True, message
            
        except Exception as e:
            message = f"Error cleaning temp files: {str(e)}"
            logger.error(message)
            return False, message
    
    def get_files_in_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of files in directory.
        
        Args:
            directory: Directory path
            extensions: Filter by extensions (e.g., ['.txt', '.json'])
            
        Returns:
            List[str]: List of file paths
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return []
            
            files = []
            for item in dir_path.iterdir():
                if item.is_file():
                    if extensions is None or item.suffix.lower() in extensions:
                        files.append(str(item))
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_directory_size(self, directory: str) -> float:
        """
        Get total size of directory in MB.
        
        Args:
            directory: Directory path
            
        Returns:
            float: Size in MB
        """
        try:
            dir_path = Path(directory)
            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
            return 0.0
    
    def create_archive(
        self,
        directory: str,
        archive_path: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Create a ZIP archive of a directory.
        
        Args:
            directory: Directory to archive
            archive_path: Output archive path (optional)
            
        Returns:
            Tuple[bool, str, str]: (success, archive_path, message)
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return False, "", f"Directory not found: {directory}"
            
            if archive_path is None:
                archive_path = f"{directory}.zip"
            
            # Ensure archive path ends with .zip
            if not archive_path.lower().endswith('.zip'):
                archive_path += '.zip'
            
            # Use zipfile directly for better control and performance
            # shutil.make_archive can be slow and sometimes includes full paths
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in dir_path.rglob('*'):
                    if file.is_file() and str(file) != archive_path:
                        # Calculate relative path for the archive
                        rel_path = file.relative_to(dir_path)
                        
                        # Determine compression method based on file type
                        # Store media files (already compressed) to save time
                        # Compress text/json files
                        compression = zipfile.ZIP_DEFLATED
                        if file.suffix.lower() in SUPPORTED_FORMATS:
                            compression = zipfile.ZIP_STORED
                            
                        zipf.write(file, rel_path, compress_type=compression)
            
            final_path = archive_path
            size_mb = Path(final_path).stat().st_size / (1024 * 1024)
            message = f"Archive created: {Path(final_path).name} ({size_mb:.2f} MB)"
            logger.info(message)
            
            return True, final_path, message
            
        except Exception as e:
            message = f"Error creating archive: {str(e)}"
            logger.error(message)
            return False, "", message


def main():
    """Test file manager functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = FileManager()
    
    print("FileManager initialized")
    print(f"Output directory: {manager.options.output_base_dir}")
    print(f"Temp directory: {manager.options.temp_dir}")
    
    # Test validation
    test_files = ["test.mp3", "test.mp4", "test.txt"]
    for file in test_files:
        valid, msg = manager.validate_audio_file(file)
        status = "✅" if valid else "❌"
        print(f"{status} {file}: {msg}")


if __name__ == "__main__":
    main()
