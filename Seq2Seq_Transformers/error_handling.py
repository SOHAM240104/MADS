"""
Custom exceptions and error handling for the command classification pipeline.
"""

from typing import Optional, Any, Dict
import traceback
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CommandClassifierError(Exception):
    """Base exception class for command classifier errors."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'error_code': self.error_code,
            'message': str(self),
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc()
        }

class DataLoadError(CommandClassifierError):
    """Raised when there are issues loading the dataset."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'DATA_LOAD_ERROR', details)

class ValidationError(CommandClassifierError):
    """Raised when data validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'VALIDATION_ERROR', details)

class ModelError(CommandClassifierError):
    """Raised when there are issues with the model."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'MODEL_ERROR', details)

class TrainingError(CommandClassifierError):
    """Raised when training encounters an error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'TRAINING_ERROR', details)

class GPUError(CommandClassifierError):
    """Raised when there are GPU-related issues."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 'GPU_ERROR', details)

def handle_error(error: Exception, 
                context: Optional[Dict[str, Any]] = None,
                raise_error: bool = True) -> Optional[Dict[str, Any]]:
    """
    Handle errors in a consistent way across the pipeline.
    
    Args:
        error: The exception that was raised
        context: Additional context about when/where the error occurred
        raise_error: Whether to re-raise the error after handling
        
    Returns:
        Dictionary with error details if raise_error is False
    """
    error_info = {
        'error_type': type(error).__name__,
        'message': str(error),
        'context': context or {},
        'timestamp': datetime.utcnow().isoformat(),
        'traceback': traceback.format_exc()
    }
    
    if isinstance(error, CommandClassifierError):
        error_info.update(error.to_dict())
    
    # Log the error
    logger.error(
        f"Error occurred: {error_info['error_type']} - {error_info['message']}",
        extra={'error_info': error_info}
    )
    
    if raise_error:
        raise
    
    return error_info

def setup_error_handling(log_file: str = 'error.log') -> None:
    """
    Set up error handling and logging configuration.
    
    Args:
        log_file: Path to the error log file
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set up custom error handler for uncaught exceptions
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Handle keyboard interrupt specially
            logger.warning("Training interrupted by user")
            return
            
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    import sys
    sys.excepthook = handle_uncaught_exception