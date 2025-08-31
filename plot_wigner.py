"""
Wigner Function Plotter
=======================

This script plots the Wigner function for a given simulation run at a specific time.

Usage:
    python plot_wigner.py --run-dir <path> --g-value <g> --time <t>
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
from qutip import wigner, Qobj
import matplotlib.pyplot as plt

from utils import setup_logging, SimulationDataFinder, find_nearest_time_index

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot Wigner function for a simulation run.',
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

def main():
    """Main function to plot the Wigner function."""
    args = parse_arguments()
    output_file = args.output_file or args.run_dir / f"wigner_g{args.g_value}_t{args.time}.png"
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

    # Trace out the qubit and cavity subsystems to get the vibrational state
    vib_state = state_to_plot.ptrace(2)

    x = np.linspace(-5, 5, 200)
    W = wigner(vib_state, x, x)

    plt.figure()
    plt.contourf(x, x, W, 100, cmap='RdBu')
    plt.colorbar()
    plt.title(f'Wigner function at t={times[time_index]:.2f} for g={args.g_value}')
    plt.xlabel('x')
    plt.ylabel('p')
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Wigner function plot saved to {output_file}")

if __name__ == "__main__":
    main()