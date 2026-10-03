"""
Logging configuration for FELsim.

Provides consistent logging setup with configurable verbosity and output options.
"""

import logging
import sys
from pathlib import Path

DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEBUG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'


def setup_logging(level=logging.INFO, log_file=None, console_output=True, format_string=None):
    """
    Configure root logger for the simulator.

    Parameters
    ----------
    level : int
        Logging level (default: INFO)
    log_file : str or Path, optional
        If provided, also log to this file
    console_output : bool
        Whether to output to console (default: True)
    format_string : str, optional
        Custom format string; if None, uses appropriate default

    Returns
    -------
    logging.Logger
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if format_string is None:
        format_string = DEBUG_FORMAT if level == logging.DEBUG else DEFAULT_FORMAT

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    if console_output:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name, debug=None):
    """
    Get a module-specific logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    debug : bool or None
        If True, force DEBUG level
        If False, force INFO level
        If None, inherit from parent logger

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if debug is True:
        logger.setLevel(logging.DEBUG)
    elif debug is False:
        logger.setLevel(logging.INFO)
    # If None, don't set level - inherit from parent

    return logger


def get_logger_with_fallback(name, debug_arg=None):
    """
    Get a logger and determine effective debug state.

    Returns both logger and the effective debug flag, useful for
    synchronizing instance debug attributes with logging state.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    debug_arg : bool or None
        Explicitly provided debug flag, or None to inherit

    Returns
    -------
    tuple: (logging.Logger, bool)
        Logger instance and effective debug state
    """
    logger = logging.getLogger(name)

    if debug_arg is True:
        logger.setLevel(logging.DEBUG)
        effective_debug = True
    elif debug_arg is False:
        logger.setLevel(logging.INFO)
        effective_debug = False
    else:
        # Inherit - determine effective level
        effective_debug = logger.getEffectiveLevel() <= logging.DEBUG

    return logger, effective_debug


def is_debug_enabled():
    """Check if root logger is at DEBUG level."""
    return logging.getLogger().level <= logging.DEBUG


def setup_for_simulation(debug=False, output_dir=None):
    """
    Quick setup for typical simulation runs.

    Parameters
    ----------
    debug : bool
        Enable verbose debug output
    output_dir : Path or str, optional
        If provided, log to 'simulation.log' in this directory
    """
    level = logging.DEBUG if debug else logging.INFO

    log_file = None
    if output_dir is not None:
        log_file = Path(output_dir) / 'simulation.log'

    return setup_logging(level=level, log_file=log_file)


def quiet_mode():
    """Suppress all logging except warnings and errors."""
    logging.getLogger().setLevel(logging.WARNING)


def set_module_level(module_name, level):
    """
    Adjust logging level for a specific module.

    Parameters
    ----------
    module_name : str
        Module name (e.g., 'beamOptimizer')
    level : int
        Logging level for this module
    """
    logging.getLogger(module_name).setLevel(level)


def level_from_debug_flag(debug):
    """Convert boolean debug flag to logging level."""
    return logging.DEBUG if debug else logging.INFO
