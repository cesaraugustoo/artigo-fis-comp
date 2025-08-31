#!/usr/bin/env python3
"""
Metric Plotter
=============

This script plots metrics at specific time points or intervals from saved simulation results.

Usage:
    python plot_metric.py --run-dir results/h_plus_classical_full --metric mean_num --g-value 0.2 --time 10.0
    python plot_metric.py --run-dir results/h_plus_classical_full --metric mean_num --g-value 0.2 --time-start 5.0 --time-end 15.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from utils import (
    setup_logging,
    find_nearest_time_index,
    find_time_indices_in_interval,
    get_interpolated_value,
    add_time_selection_args
)
from metrics import get_calculator
from config import MetricsConfig

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Metric Plotter for Quantum Simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--run-dir', type=Path, required=True,
                       help='Path to simulation run directory')
    parser.add_argument('--metric', type=str, required=True,
                       help='Name of the metric to plot')
    parser.add_argument('--g-value', type=float, required=True,
                       help='Coupling strength value')

    # Output options
    parser.add_argument('--output-file', type=Path,
                       help='Path to save the plot (default: auto-generated)')

    # Add time selection arguments
    add_time_selection_args(parser)

    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')

    return parser.parse_args()

def load_metric_data(run_dir: Path, metric: str, g_value: float):
    """Load metric data from file.

    Args:
        run_dir: Path to simulation run directory
        metric: Name of the metric
        g_value: Coupling strength value

    Returns:
        Tuple of (times, metric_values)
    """
    # Find metric file
    metric_dir = run_dir / 'metrics' / metric

    # Try different naming patterns
    patterns = [
        f'*_g{g_value}_*.npz',  # Pattern with g-value in filename
        f'*_values.npz'          # Generic pattern for values file
    ]

    metric_files = []
    for pattern in patterns:
        files = list(metric_dir.glob(pattern))
        if files:
            metric_files.extend(files)
            break

    if not metric_files:
        raise FileNotFoundError(f"No metric file found for g={g_value} in {metric_dir}")

    # Load metric data
    metric_data = np.load(metric_files[0])

    # Try different key patterns
    for key in metric_data.keys():
        # Check if this key contains data for our g-value
        if f'g{g_value}' in key or f'g_{g_value}' in key:
            data = metric_data[key]
            times = data[:, 0]
            if data.shape[1] == 2:
                metric_values = data[:, 1]
            else:
                metric_values = data[:, 1:]
            return times, metric_values

    # If no specific g-value key found, try generic keys
    if len(metric_data.keys()) == 1:
        # If there's only one key, use it
        key = list(metric_data.keys())[0]
        data = metric_data[key]
        times = data[:, 0]
        if data.shape[1] == 2:
            metric_values = data[:, 1]
        else:
            metric_values = data[:, 1:]
        return times, metric_values

    # If multiple keys, look for one that might match our g-value
    for key in metric_data.keys():
        # Try to extract g-value from the key
        import re
        match = re.search(r'g[_]?(\d+\.\d+)', key)
        if match and float(match.group(1)) == g_value:
            data = metric_data[key]
            times = data[:, 0]
            if data.shape[1] == 2:
                metric_values = data[:, 1]
            else:
                metric_values = data[:, 1:]
            return times, metric_values

    raise ValueError(f"No data found for g={g_value} in {metric_files[0]}")

def load_config(run_dir: Path):
    """Load simulation configuration.

    Args:
        run_dir: Path to simulation run directory

    Returns:
        Simulation configuration dictionary
    """
    config_file = run_dir / 'simulation_config.yaml'

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for metric plotting."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(Path('plot_metric.log'), args.debug)

    try:
        # Load simulation configuration
        config = load_config(args.run_dir)

        # Load metric data
        times, metric_values = load_metric_data(args.run_dir, args.metric, args.g_value)

        # Validate time range
        t_max = config.get('t_max', 40.0)

        if args.time is not None:
            if args.time < 0 or args.time > t_max:
                logger.warning(f"Requested time {args.time} is outside simulation range [0, {t_max}]")

            # Find nearest time index
            idx = find_nearest_time_index(times, args.time)

            # Get values at nearest time
            t_nearest = times[idx]

            if len(metric_values.shape) == 1:
                v_nearest = metric_values[idx]

                # Print values
                print(f"Time: {t_nearest:.4f}")
                print(f"Value: {v_nearest:.6f}")

                if args.interpolation and t_nearest != args.time:
                    v_interp = get_interpolated_value(times, metric_values, args.time)
                    print(f"Interpolated value at t={args.time:.4f}: {v_interp:.6f}")
            else:
                v_nearest = metric_values[idx, :]

                # Print values
                print(f"Time: {t_nearest:.4f}")
                for i, val in enumerate(v_nearest):
                    print(f"Component {i}: {val:.6f}")

                if args.interpolation and t_nearest != args.time:
                    v_interp = np.array([get_interpolated_value(times, metric_values[:, i], args.time)
                                        for i in range(metric_values.shape[1])])
                    print(f"Interpolated values at t={args.time:.4f}:")
                    for i, val in enumerate(v_interp):
                        print(f"Component {i}: {val:.6f}")

            # No plot for single time point

        elif args.time_start is not None:
            # Validate time range
            if args.time_end is None:
                args.time_end = t_max

            if args.time_start < 0 or args.time_start > t_max:
                logger.warning(f"Start time {args.time_start} is outside simulation range [0, {t_max}]")
            if args.time_end < 0 or args.time_end > t_max:
                logger.warning(f"End time {args.time_end} is outside simulation range [0, {t_max}]")
            if args.time_start >= args.time_end:
                raise ValueError(f"Start time {args.time_start} must be less than end time {args.time_end}")

            # Find time indices in interval
            indices = find_time_indices_in_interval(times, args.time_start, args.time_end)

            if len(indices) == 0:
                raise ValueError(f"No time points found in interval [{args.time_start}, {args.time_end}]")

            # Get values in interval
            t_interval = times[indices]
            v_interval = metric_values[indices]

            # Create plot
            plt.figure(figsize=(10, 6))

            if len(metric_values.shape) == 1:
                plt.plot(t_interval, v_interval)
                plt.ylabel(f'{args.metric.capitalize()} Value')
            else:
                for i in range(v_interval.shape[1]):
                    plt.plot(t_interval, v_interval[:, i], label=f'Component {i}')
                plt.ylabel(f'{args.metric.capitalize()} Values')
                plt.legend()

            plt.xlabel('Time' + (' (gt)' if config.get('use_gt_scale', False) else ' (νt)'))
            plt.title(f'{args.metric.capitalize()} Evolution for g={args.g_value}')
            plt.grid(True)

            # Save plot
            if args.output_file:
                plt.savefig(args.output_file)
                logger.info(f"Plot saved to {args.output_file}")
            else:
                output_file = args.run_dir / 'metrics' / args.metric / f'{args.metric}_g{args.g_value}_interval.png'
                plt.savefig(output_file)
                logger.info(f"Plot saved to {output_file}")

            plt.close()

        else:
            # Plot full time range
            plt.figure(figsize=(10, 6))

            if len(metric_values.shape) == 1:
                plt.plot(times, metric_values)
                plt.ylabel(f'{args.metric.capitalize()} Value')
            else:
                for i in range(metric_values.shape[1]):
                    plt.plot(times, metric_values[:, i], label=f'Component {i}')
                plt.ylabel(f'{args.metric.capitalize()} Values')
                plt.legend()

            plt.xlabel('Time' + (' (gt)' if config.get('use_gt_scale', False) else ' (νt)'))
            plt.title(f'{args.metric.capitalize()} Evolution for g={args.g_value}')
            plt.grid(True)

            # Save plot
            if args.output_file:
                plt.savefig(args.output_file)
                logger.info(f"Plot saved to {args.output_file}")
            else:
                output_file = args.run_dir / 'metrics' / args.metric / f'{args.metric}_g{args.g_value}_full.png'
                plt.savefig(output_file)
                logger.info(f"Plot saved to {output_file}")

            plt.close()

    except Exception as e:
        logger.error(f"Error plotting metric: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
