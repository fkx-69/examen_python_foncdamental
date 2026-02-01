"""
Utilities and Decorators (Decorator Pattern).
"""

import time
import functools
import logging

# Basic logging configuration if not already configured
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def timing_decorator(func):
    """Decorator measuring function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[TIMING] {func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper

def logging_decorator(func):
    """Decorator adding logs before and after execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Starting execution of: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Finished execution of: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise e
    return wrapper
