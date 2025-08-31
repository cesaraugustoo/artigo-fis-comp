"""
Photon Distribution Plotter
=========================

This script plots the photon number distribution for a given simulation run at a specific time.

Usage:
    python plot_photon_distribution.py --run-dir <path> --g-value <g> --time <t>
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from utils import setup_logging, SimulationDataFinder, find_nearest_time_index

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot photon distribution for a simulation run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--run-dir', type=Path, required=True,
                       help='Path to the simulation output directory')
    parser.add_argument('--g-value', type=float, required=True,
                       help='The g-value to plot')
    parser.add_argument('--time', type=float, required=True,
                       help='The time point to plot')
    parser.add_argument('--output-file', type=Path,
                       help='Path to save the plot file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    return parser.parse_args()

def calculate_photon_distribution(state: Qobj, N_b: int):
    """Calculate photon number distribution from quantum state."""
    # Trace out other subsystems to get cavity state
    rho_cavity = state.ptrace(1)  # Assuming cavity is the second subsystem (index 1)

    # Get diagonal elements (photon number probabilities)
    probs = np.real(rho_cavity.diag())

    # Calculate mean photon number
    n_values = np.arange(N_b)
    mean = np.sum(n_values * probs)

    # Calculate variance
    variance = np.sum((n_values - mean)**2 * probs)

    # Calculate R parameter (sub-Poissonian statistics)
    if mean > 0:
        R = 1 - variance / mean
    else:
        R = 0

    return probs, mean, variance, R

def plot_distribution(probs, mean, variance, R, time_point, g_value, output_file):
    """Plot photon number distribution."""
    N_b = len(probs)
    n_values = np.arange(N_b)

    plt.figure(figsize=(10, 6))
    plt.bar(n_values, probs, color='blue', alpha=0.7)
    plt.xlabel('Photon Number n')
    plt.ylabel('Probability P(n)')
    plt.title(f'Photon Distribution at t={time_point:.2f} for g={g_value}\nMean={mean:.2f}, Var={variance:.2f}, R={R:.3f}')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Photon distribution plot saved to {output_file}")

def main():
    """Main function to plot the photon distribution."""
    args = parse_arguments()
    output_file = args.output_file or args.run_dir / f"photon_dist_g{args.g_value}_t{args.time}.png"
    setup_logging(output_file.with_suffix('.log'), args.debug)

    try:
        finder = SimulationDataFinder(args.run_dir)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    evolution = finder.find_and_load_evolution(args.g_value)
    if not evolution:
        logger.error(f"Could not find evolution data for g={args.g_value}")
        sys.exit(1)

    times, states = evolution
    time_index = find_nearest_time_index(times, args.time)
    state_to_plot = states[time_index]

    probs, mean, var, R = calculate_photon_distribution(state_to_plot, finder.config.N_b)
    plot_distribution(probs, mean, var, R, times[time_index], args.g_value, output_file)

if __name__ == "__main__":
    main()