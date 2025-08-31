"""
Utilities Module
==============

This module provides utility functions used across the quantum simulation project.
It includes functions for logging setup, memory management, plotting utilities,
and various helper functions.

Functions:
    setup_logging: Configure logging for the project
    clear_matplotlib: Clear matplotlib memory
    create_output_dirs: Create required output directories
    format_filename: Format filenames consistently
    memory_usage: Get current memory usage
    time_evolution: Time evolution wrapper with progress bar
"""

import gc
import logging
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from functools import wraps
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, qload
from tqdm import tqdm
import psutil

from config import SimulationConfig

class SimulationDataFinder:
    """
    Utility class to find and load simulation data from a run directory.
    """
    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        self.config = self._load_config()
        self.states_dir = self.run_dir / 'states'

    def _load_config(self) -> SimulationConfig:
        """Loads the simulation configuration."""
        config_path = self.run_dir / 'simulation_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"simulation_config.yaml not found in {self.run_dir}")
        return SimulationConfig.from_yaml(config_path)

    def find_and_load_evolution(self, g_value: float) -> Optional[Tuple[np.ndarray, List[Qobj]]]:
        """Finds and loads the time and state evolution files for a given g."""
        rwa_suffix = '_rwa' if self.config.rwa else '_full'
        base_filename = f"{self.config.N_a}_{self.config.hamiltonian}_{self.config.initial_state}{rwa_suffix}"

        time_file = self.states_dir / f"{base_filename}_times_g{g_value}.npy"
        state_file = self.states_dir / f"{base_filename}_states_g{g_value}.qu"

        if time_file.exists() and state_file.exists():
            times = np.load(time_file)
            states = qload(str(state_file).replace('.qu', ''))
            return times, states

        logging.warning(f"Could not find evolution data for g={g_value} in {self.run_dir}")
        return None

def setup_logging(log_file: Optional[Path] = None, debug: bool = False) -> None:
    """Configure logging for the project.

    Args:
        log_file: Optional log file path (Path object)
        debug: Enable debug logging

    Example:
        >>> setup_logging(log_file=Path('simulation.log'), debug=True)
    """
    # Set log level
    log_level = logging.DEBUG if debug else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Ensure the parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.info("Logging configured successfully")

def clear_matplotlib() -> None:
    """Clear matplotlib memory and close all figures.

    This function should be called after generating plots to free memory.
    """
    plt.close('all')
    gc.collect()

def create_output_dirs(base_dir: Path, subdirs: list) -> Dict[str, Path]:
    """Create output directories for simulation results.

    This function creates a directory structure for simulation outputs at the project root level,
    separate from the source code directory. It handles both the base output directory and
    any required subdirectories for different types of outputs (states, metrics, etc.).

    Args:
        base_dir: Base output directory (should be at project root level)
        subdirs: List of subdirectory names to create

    Returns:
        Dictionary mapping subdirectory names to Path objects

    Example:
        >>> # Creating standard simulation directories
        >>> dirs = create_output_dirs(
        ...     Path('article_code/results/sim_20240118'),
        ...     ['states', 'metrics', 'plots', 'logs']
        ... )
        >>> print(dirs['metrics'])  # Access metrics directory path
    """
    paths = {}
    try:
        # Ensure base_dir is absolute and at project root level
        base_dir = base_dir.resolve()

        # Create base directory
        base_dir.mkdir(parents=True, exist_ok=True)
        paths['base'] = base_dir

        # Create subdirectories
        for subdir in subdirs:
            path = base_dir / subdir
            path.mkdir(exist_ok=True)
            paths[subdir] = path

        # Log directory creation
        logging.info(f"Created output directory structure at: {base_dir}")
        for subdir, path in paths.items():
            logging.debug(f"Created {subdir} directory: {path}")

        return paths

    except Exception as e:
        logging.error(f"Failed to create directories: {str(e)}")
        raise

def generate_simulation_dirname(config: SimulationConfig) -> str:
    """Generate a descriptive directory name for a simulation run.

    Args:
        config: Simulation configuration object

    Returns:
        Formatted directory name for the simulation
    """
    # Format timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create descriptive name using key parameters
    name_parts = [
        f"sim",  # Prefix
        f"Na{config.N_a}",
        f"Nb{config.N_b}",
        f"h{config.hamiltonian.split('_')[1]}",  # Convert h_plus to hplus
        config.initial_state
    ]

    # Add loss parameters if present
    if config.kappa is not None:
        name_parts.append(f"κ{config.kappa:.3f}")
    if config.gamma is not None:
        name_parts.append(f"γ{config.gamma:.3f}")
    if config.gamma_sp is not None:
        name_parts.append(f"Γ{config.gamma_sp:.3f}")

    # Add timestamp
    name_parts.append(timestamp)

    return "_".join(name_parts)

def format_filename(base_name: str, params: Dict[str, Any], ext: str) -> str:
    """Format filename with parameters.

    Args:
        base_name: Base filename
        params: Dictionary of parameters to include
        ext: File extension

    Returns:
        Formatted filename

    Example:
        >>> params = {'N': 10, 'g': 0.5}
        >>> format_filename('sim', params, 'dat')
        'sim_N10_g0.5.dat'
    """
    # Format parameters as strings
    param_strs = [f"{k}{v}" for k, v in params.items()]

    # Join with underscores and add extension
    return f"{base_name}_{'_'.join(param_strs)}.{ext}"

def memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary containing memory usage statistics in MB

    Example:
        >>> mem = memory_usage()
        >>> print(f"Used: {mem['used']:.1f} MB")
    """
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'used': psutil.virtual_memory().used / 1024 / 1024  # System memory used
    }

def time_evolution(times: np.ndarray):
    """Decorator for time evolution calculations with progress bar.

    Args:
        times: Array of time points

    Example:
        >>> @time_evolution(times)
        >>> def evolve_state(state, hamiltonian):
        >>>     # Evolution implementation
        >>>     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            with tqdm(total=len(times), desc="Evolution") as pbar:
                for t in times:
                    result = func(t, *args, **kwargs)
                    results.append(result)
                    pbar.update(1)
            return results
        return wrapper
    return decorator

def timing_decorator(func):
    """Decorator to measure execution time of functions.

    Example:
        >>> @timing_decorator
        >>> def long_calculation():
        >>>     # Calculation implementation
        >>>     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")

        return result
    return wrapper

def validate_parameters(validation_dict: Dict[str, Any]):
    """Decorator for parameter validation.

    Args:
        validation_dict: Dictionary mapping parameter names to validation functions

    Example:
        >>> def positive(x): return x > 0
        >>> @validate_parameters({'N': positive, 'g': positive})
        >>> def simulate(N, g):
        >>>     # Simulation implementation
        >>>     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Combine args and kwargs
            bound_args = func.__annotations__.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate parameters
            for param_name, validate in validation_dict.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validate(value):
                        raise ValueError(
                            f"Parameter {param_name} failed validation with value {value}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator

def save_configuration(config: Dict[str, Any], filename: str) -> None:
    """Save configuration to file with timestamp.

    Args:
        config: Configuration dictionary
        filename: Output filename

    Example:
        >>> config = {'N': 10, 'g': 0.5}
        >>> save_configuration(config, 'sim_config.txt')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, 'w') as f:
        f.write(f"Configuration saved at: {timestamp}\n\n")
        for key, value in sorted(config.items()):
            f.write(f"{key}: {value}\n")

def load_data(filename: Path) -> np.ndarray:
    """Load numerical data from file with error handling.

    Args:
        filename: Path to data file

    Returns:
        Numpy array containing loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not filename.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        return np.load(filename)
    except Exception as e:
        raise ValueError(f"Failed to load data from {filename}: {str(e)}")

def find_nearest_time_index(times: np.ndarray, target_time: float) -> int:
    """Find the index of the time step closest to target_time.

    Args:
        times: Array of time points
        target_time: Target time to find

    Returns:
        Index of the closest time point
    """
    return np.abs(times - target_time).argmin()

def find_time_indices_in_interval(times: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    """Find indices of time steps within the [start_time, end_time] interval (inclusive).

    Args:
        times: Array of time points
        start_time: Start of time interval
        end_time: End of time interval

    Returns:
        Array of indices within the interval
    """
    return np.where((times >= start_time) & (times <= end_time))[0]

def get_interpolated_value(times: np.ndarray, values: np.ndarray, target_time: float) -> Union[float, np.ndarray]:
    """Perform linear interpolation to estimate the metric value at target_time.

    Args:
        times: Array of time points
        values: Array of values corresponding to time points
        target_time: Target time for interpolation

    Returns:
        Interpolated value at target_time
    """
    if target_time <= times[0]:
        return values[0]
    if target_time >= times[-1]:
        return values[-1]

    # Find the two closest time points
    idx = np.searchsorted(times, target_time)
    t0, t1 = times[idx-1], times[idx]
    v0, v1 = values[idx-1], values[idx]

    # Linear interpolation
    return v0 + (v1 - v0) * (target_time - t0) / (t1 - t0)

def add_time_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add time selection arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--time', type=float,
                          help='Specific time point for analysis')
    time_group.add_argument('--time-start', type=float,
                          help='Start time for analysis interval')

    # Only add time-end if time-start is provided
    parser.add_argument('--time-end', type=float,
                       help='End time for analysis interval (requires --time-start)')

    parser.add_argument('--interpolation', action='store_true',
                       help='Use linear interpolation for time points between steps')

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        """Initialize timer context manager.

        Args:
            name: Name of the operation being timed (unused if not "Full simulation")
        """
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        # Only log for non-simulation timing to avoid redundancy
        if self.name != "Full simulation":
            logging.info(f"{self.name} completed in {elapsed_time:.2f} seconds")
