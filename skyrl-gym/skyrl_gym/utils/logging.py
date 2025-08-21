"""
Simple logging utility for SkyRL Gym.
"""
import logging
import sys
import os

def get_logger(name: str):
    """Get a simple logger for the given module name."""
    logger = logging.getLogger(f"skyrl_gym.{name}")
    
    if not logger.handlers:
        # Get log level from environment or default to INFO
        level = os.environ.get("SKYRL_LOG_LEVEL", "INFO").upper()
        
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
    
    return logger