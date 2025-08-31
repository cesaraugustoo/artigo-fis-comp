#!/usr/bin/env python3
"""
Batch Metrics Calculator
=======================

This script calculates metrics for multiple simulation runs in batch mode.
It loads previously saved quantum states from simulation output directories
and calculates the specified metric for each run.

Usage:
    python batch_calculate_metrics.py --run-dirs dir1 dir2 --metric wigner_neg
    python batch_calculate_metrics.py --run-dirs results/sim_* --metric coherence --g-values 0.2 2.0

Arguments:
    --run-dirs: List of paths to simulation run directories
    --metric: Name of the metric to calculate (e.g., 'wigner_neg', 'coherence')
    --g-values: (Optional) Specific g/ν values to process
    --log-file: (Optional) Path to log file (default: batch_metrics.log)
    --debug: (Optional) Enable debug logging
"""

import gc
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qutip import qload

from config import SimulationConfig
from metrics import get_calculator, available_metrics
from operators import QuantumOperators
from utils import setup_logging, clear_matplotlib, SimulationDataFinder

# Setup logging
logger = logging.getLogger(__name__)

def _get_base_filename(config: Dict) -> str:
    """Generate base filename from configuration.

    Args:
        config: Dictionary of simulation parameters

    Returns:
        Base filename for simulation files
    """
    # Extract parameters for filename
    hamiltonian = config['hamiltonian']
    initial_state = config['initial_state']

    # Create base filename
    return f"{hamiltonian}_{initial_state}"

def _get_metric_results_filepath(metrics_output_dir: Path, base_filename: str, metric_name: str, rwa_flag: bool) -> Path:
    """Get path for metric results file.

    Args:
        metrics_output_dir: Path to metrics output directory
        base_filename: Base filename for simulation files
        metric_name: Name of the metric
        rwa_flag: Whether to use RWA files

    Returns:
        Path to metric results file
    """
    # Determine file type (rwa or full)
    file_type = 'rwa' if rwa_flag else 'full'

    # Create filename
    filename = f"{base_filename}_{file_type}_{metric_name}_values.npz"

    return metrics_output_dir / filename

def calculate_metrics_for_g_value(
    finder: SimulationDataFinder,
    g_value: float,
    metric_calculator,
    params: Dict
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Calculate metrics for a specific g-value.

    Args:
        finder: SimulationDataFinder instance
        g_value: Coupling strength value
        metric_calculator: Metric calculator instance
        params: Dictionary of simulation parameters

    Returns:
        Tuple of (times, metric_values)

    Raises:
        FileNotFoundError: If state or time files are not found
        Exception: If metric calculation fails
    """
    try:
        # Get state and time files
        evolution = finder.find_and_load_evolution(g_value)
        if not evolution:
            return None

        times, states = evolution

        # Calculate metric values
        logger.info(f"Calculating {params['metric']} for g={g_value}...")
        metric_values = metric_calculator.calculate(states, times, g_value)

        # Clean up
        del states
        gc.collect()
        clear_matplotlib()

        return times, metric_values

    except Exception as e:
        logger.error(f"Error calculating metrics for g={g_value}: {str(e)}")
        raise

def save_metric_results(results: Dict[float, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Save calculated metrics to file.

    Args:
        results: Dictionary mapping g-values to (times, metric_values) tuples
        output_path: Path to save results

    Raises:
        IOError: If saving fails
    """
    try:
        # Create a dictionary of arrays to save
        save_dict = {}

        for g_value, (times, metric_values) in results.items():
            # Create key for this g-value
            save_key = f'g_{g_value}_data'
            logger.info(f"Preparing data for g={g_value} with key {save_key}")

            # Handle different shapes of metric_values
            if not hasattr(metric_values, 'shape'):
                # Convert scalar to array
                metric_values = np.array([metric_values])

            if len(metric_values.shape) == 1:
                # If metric_values is 1D, stack times and values side by side
                logger.info(f"Stacking 1D metric values with times")
                save_dict[save_key] = np.column_stack((times, metric_values))
            elif metric_values.shape[1] == 1:
                # If metric_values is 2D but with only one column, flatten it
                logger.info(f"Flattening 2D metric values with single column")
                save_dict[save_key] = np.column_stack((times, metric_values.flatten()))
            else:
                # If metric_values is 2D with multiple columns, stack times and all values
                logger.info(f"Stacking 2D metric values with times")
                save_dict[save_key] = np.column_stack((times.reshape(-1, 1), metric_values))

        # Save the data
        np.savez(output_path, **save_dict)
        logger.info(f"Saved metric data to: {output_path}")

    except Exception as e:
        logger.error(f"Error saving metric results: {str(e)}")
        raise IOError(f"Failed to save metric results: {str(e)}")

def process_run_directory(run_dir: Path, metric_name: str, target_g_values: List[float] = None) -> bool:
    """Process a single run directory.

    Args:
        run_dir: Path to simulation run directory
        metric_name: Name of the metric to calculate
        target_g_values: Specific g-values to process (if None, use all from config)

    Returns:
        True if processing was successful, False otherwise
    """
    logger.info(f"Processing directory: {run_dir}")

    try:
        # Load configuration
        finder = SimulationDataFinder(run_dir)
        params = finder.config.to_dict()

        # Set metric name in params
        params['metric'] = metric_name

        # Get dimensions for operators
        N_a = params['N_a']
        N_b = params['N_b']

        # Initialize operators
        operators = QuantumOperators(N_a, N_b)
        operators.create_basic_operators()
        operators.create_composite_operators()

        # Get metric calculator
        metric_calculator = get_calculator(metric_name)

        # Set operators for metric calculator if it has the method
        if hasattr(metric_calculator, 'initialize_operators'):
            metric_calculator.initialize_operators(N_a, N_b)

        # Determine g-values to process
        config_g_values = params['g_values']
        g_values_to_process = target_g_values if target_g_values else config_g_values

        logger.info(f"Processing g-values: {g_values_to_process}")

        # Setup output directory for metrics
        metrics_output_dir = Path(run_dir) / 'metrics' / metric_name
        metrics_output_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename
        base_filename = _get_base_filename(params)

        # Get output file path
        output_path = _get_metric_results_filepath(
            metrics_output_dir, base_filename, metric_name, params['rwa'])

        # Check if output file already exists
        if output_path.is_file():
            logger.info(f"Skipping {run_dir}: Metric results already exist at {output_path}")
            return True

        # Process each g-value
        results = {}

        for g_value in g_values_to_process:
            try:
                # Calculate metrics for this g-value
                result = calculate_metrics_for_g_value(
                    finder, g_value, metric_calculator, params)

                if result:
                    times, metric_values = result
                    # Store results
                    results[g_value] = (times, metric_values)

            except Exception as e:
                logger.error(f"Error processing g={g_value}: {str(e)}")
                # Continue with next g-value
                continue

        # Save results if any were calculated
        if results:
            save_metric_results(results, output_path)
            return True
        else:
            logger.warning(f"No results calculated for {run_dir}")
            return False

    except Exception as e:
        logger.error(f"Error processing directory {run_dir}: {str(e)}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Batch Metrics Calculator for Quantum Simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--run-dirs', type=str, nargs='+', required=True,
                       help='List of paths to simulation run directories')
    parser.add_argument('--metric', type=str, required=True,
                       choices=available_metrics(),
                       help='Name of the metric to calculate')

    # Optional arguments
    parser.add_argument('--g-values', type=float, nargs='*',
                       help='Specific g/ν values to process (default: all in config)')
    parser.add_argument('--log-file', type=str, default='batch_metrics.log',
                       help='Path to log file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    return parser.parse_args()

def main() -> None:
    """Main entry point for batch metrics calculation."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    log_file = Path(args.log_file)
    setup_logging(log_file, args.debug)

    logger.info("Starting batch metrics calculation")
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Run directories: {args.run_dirs}")
    if args.g_values:
        logger.info(f"Target g-values: {args.g_values}")

    # Process each run directory
    success_count = 0
    failure_count = 0

    for run_dir_str in args.run_dirs:
        # Convert to Path object
        run_dir = Path(run_dir_str)

        # Check if directory exists
        if not run_dir.is_dir():
            logger.error(f"Directory not found: {run_dir}")
            failure_count += 1
            continue

        # Process directory
        success = process_run_directory(run_dir, args.metric, args.g_values)

        if success:
            success_count += 1
        else:
            failure_count += 1

    # Log summary
    logger.info("Batch processing complete")
    logger.info(f"Successful runs: {success_count}")
    logger.info(f"Failed runs: {failure_count}")

    # Return appropriate exit code
    if failure_count > 0:
        logger.warning(f"Some runs failed ({failure_count}/{success_count + failure_count})")
        sys.exit(1)
    else:
        logger.info("All runs completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
