"""

Compare Runs Module
===================

This module compares simulation results from two different runs.

Usage:
    python compare_runs.py --dir1 <path1> --dir2 <path2> --compare fidelity mean_num --plot
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
from qutip import fidelity
import matplotlib.pyplot as plt

from utils import setup_logging, SimulationDataFinder

# TODO: This is a temporary solution. The SimulationDataFinder class should be
# updated in utils.py to include the find_and_load_metric method.

def find_and_load_metric(self, metric_name: str, g_value: float):
    """Finds and loads the metric data file for a given g."""
    rwa_suffix = '_rwa' if self.config.rwa else '_full'
    base_filename = f"{self.config.N_a}_{self.config.hamiltonian}_{self.config.initial_state}"
    metric_dir = self.run_dir / 'metrics' / metric_name
    metric_file = metric_dir / f"{base_filename}{rwa_suffix}_{metric_name}_values.npz"

    if metric_file.exists():
        data = np.load(metric_file)
        return data[f"g_{g_value}_times"], data[f"g_{g_value}_values"]

    logging.warning(f"Could not find metric data for g={g_value} in {self.run_dir}")
    return None, None

SimulationDataFinder.find_and_load_metric = find_and_load_metric

# Setup logging
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare two simulation runs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dir1', type=Path, required=True,
                       help='Path to the first simulation output directory')
    parser.add_argument('--dir2', type=Path, required=True,
                       help='Path to the second simulation output directory')
    parser.add_argument('--compare', type=str, nargs='+', default=['fidelity'],
                       choices=['fidelity', 'mean_num'],
                       help='Type of comparison to perform')
    parser.add_argument('--output-dir', type=Path,
                       help='Directory to save comparison results and plots')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for comparison results')
    parser.add_argument('--g-values', type=float, nargs='+',
                       help='Specific g-values to compare (optional)')
    parser.add_argument('--legend-labels', type=str, nargs=2, default=['Run 1', 'Run 2'],
                       help='Labels for the legend in comparison plots')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    return parser.parse_args()


def _compare_fidelity(data1, data2, g_value):
    """Compare fidelity between two simulation runs."""
    times1, states1 = data1
    times2, states2 = data2

    if not np.allclose(times1, times2):
        logger.warning(f"Time points for g={g_value} do not match, skipping fidelity comparison.")
        return None, None

    fidelities = [fidelity(state1, state2) for state1, state2 in zip(states1, states2)]
    return times1, np.array(fidelities)


def _compare_mean_number(finder1, finder2, g_value):
    """Compare mean photon number between two simulation runs."""
    try:
        times1, mean_num1 = finder1.find_and_load_metric('mean_num', g_value)
        times2, mean_num2 = finder2.find_and_load_metric('mean_num', g_value)
    except FileNotFoundError:
        logger.warning(f"Mean number data not found for g={g_value}, skipping.")
        return None, None, None

    if times1 is None or times2 is None:
        return None, None, None

    if not np.allclose(times1, times2):
        logger.warning(f"Time points for g={g_value} do not match, skipping mean number comparison.")
        return None, None, None

    return times1, mean_num1, mean_num2


def _plot_mean_number_comparison(times, mean_num1, mean_num2, g_value, labels, output_dir):
    """Plot the mean number comparison for each subsystem."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    subsystem_names = ['Qubit <σz>', 'Cavity <n_b>', 'Vibrational <n_a>']

    for i, ax in enumerate(axes):
        ax.plot(times, mean_num1[:, i], label=labels[0])
        ax.plot(times, mean_num2[:, i], label=labels[1], linestyle='--')
        ax.set_ylabel(subsystem_names[i])
        ax.grid(True)

    axes[-1].set_xlabel('Time')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(1.1, 0.9))
    plt.suptitle(f'Mean Number Comparison (g={g_value})')
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = output_dir / f"mean_num_comparison_g{g_value}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Mean number comparison plot saved to {plot_path}")


def main():
    """Main function to run the comparator."""
    args = parse_arguments()
    output_dir = args.output_dir or Path.cwd() / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / 'comparison.log', args.debug)

    try:
        finder1 = SimulationDataFinder(args.dir1)
        finder2 = SimulationDataFinder(args.dir2)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    g_values = args.g_values or finder1.config.g_values

    if not g_values:
        logger.error("Could not determine g-values from config.")
        sys.exit(1)

    for g in g_values:
        logger.info(f"Comparing results for g = {g}")

        if 'fidelity' in args.compare:
            evolution1 = finder1.find_and_load_evolution(g)
            evolution2 = finder2.find_and_load_evolution(g)

            if not evolution1 or not evolution2:
                logger.warning(f"Skipping fidelity for g = {g} due to missing evolution data.")
                continue

            times, fidelities = _compare_fidelity(evolution1, evolution2, g)
            if times is not None and args.plot:
                plt.figure()
                plt.plot(times, fidelities)
                plt.title(f'Fidelity between runs (g={g})')
                plt.xlabel('Time')
                plt.ylabel('Fidelity')
                plt.grid(True)
                plot_path = output_dir / f"fidelity_g{g}.png"
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Fidelity plot saved to {plot_path}")

        if 'mean_num' in args.compare:
            times, mean_num1, mean_num2 = _compare_mean_number(finder1, finder2, g)
            if times is not None and args.plot:
                _plot_mean_number_comparison(times, mean_num1, mean_num2, g, args.legend_labels, output_dir)


if __name__ == "__main__":
    main()