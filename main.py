"""
Main Application Module
=====================

This module serves as the entry point for the quantum simulation project.
It handles command-line argument parsing, configuration setup, and
simulation execution.

Usage:
    python main.py --Na 20 --Nb 20 -H h_plus --initial-state classical
"""

import sys
import argparse
import cProfile
import pstats
import logging
from pathlib import Path

import numpy as np
from qutip import qsave

from config import SimulationConfig
from simulator import QuantumSimulator
from exceptions import QuantumSimError
from metrics import available_metrics
from utils import (
    setup_logging,
    Timer,
    create_output_dirs,
    generate_simulation_dirname
)

# Setup logging
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Trapped Ion-Cavity Quantum System Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hilbert space dimensions
    parser.add_argument('--Na', type=int, default=20,
                       help='Dimension for vibrational mode')
    parser.add_argument('--Nb', type=int, default=20,
                       help='Dimension for cavity mode')

    # System configuration
    parser.add_argument('-H', '--hamiltonian', type=str, required=True,
                       choices=['h_plus', 'h_minus'],
                       help='Hamiltonian type')

    # State configuration
    state_group = parser.add_argument_group('State Configuration')
    state_group.add_argument('--initial-state', type=str, default='classical',
                           choices=['classical', 'quantum', 'dressed_plus', 'dressed_minus', 'cat', 'pure_fock'],
                           help='Type of initial state')

    # Classical state parameters
    state_group.add_argument('--beta', type=float, default=1.0,
                           help='Coherent state amplitude for classical state')
    state_group.add_argument('--thermal-n', type=float, default=2.0,
                           help='Thermal occupation number for classical state')

    # Quantum state parameters
    state_group.add_argument('--cavity-n', type=int, default=1,
                           help='Cavity photon number for quantum state')
    state_group.add_argument('--vib-n', type=int, default=2,
                           help='Vibrational excitation number for quantum state')

    # Dressed state parameters
    state_group.add_argument('--cavity-state', type=str,
                           choices=['fock', 'coherent'],
                           help='Type of cavity state for dressed states')
    state_group.add_argument('--cavity-param', type=float,
                           help='Parameter for cavity state (Fock number or coherent amplitude)')

    # Vibrational state parameters
    state_group.add_argument('--vib-state', type=str,
                           choices=['fock', 'thermal'],
                           help='Type of vibrational state')
    state_group.add_argument('--vib-param', type=float,
                           help='Parameter for vibrational state (Fock number or thermal occupation)')

    # Cat state parameters
    state_group.add_argument('--alpha', type=float,
                           help='Amplitude for cat states')
    state_group.add_argument('--cat-parity', type=int, choices=[1, -1],
                           help='Parity for cat states (+1: even, -1: odd)')
    state_group.add_argument('--qubit-state', type=str,
                           choices=['e', 'g', '+', '-'],
                           help='Initial qubit state (for cat and pure_fock states)')

    # Physical parameters
    parser.add_argument('--nu', type=float, default=1.0,
                       help='Frequency parameter')
    parser.add_argument('--eta', type=float, default=0.2,
                       help='Coupling strength parameter')

    # Loss parameters (optional)
    parser.add_argument('--kappa', type=float,
                       help='Cavity decay rate (in units of ν)')
    parser.add_argument('--gamma', type=float,
                       help='Qubit dephasing rate (in units of ν)')
    parser.add_argument('--gamma-sp', type=float, default=None,
                       help='Atomic spontaneous decay rate (in units of ν). '
                            'Typical value: γ_sp/ν ≈ 0.0241')

    # Evolution parameters
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--gt-scale', action='store_true',
                          help='Use gt as time scale')
    time_group.add_argument('--nu-t-scale', action='store_true',
                          help='Use νt as time scale (default)')
    parser.add_argument('--t-max', type=float, default=40.0,
                       help='Maximum time in chosen scale')

    parser.add_argument('--n-steps', type=int, default=200,
                       help='Number of time steps')
    parser.add_argument('--g-values', type=float, nargs='+',
                       default=[0.2, 2.0],
                       help='List of coupling strengths g/nu to simulate')

    # Simulation options
    parser.add_argument('--rwa', action='store_true',
                       help='Use Rotating Wave Approximation')

    parser.add_argument('--metric', type=str,
                       choices=available_metrics(),
                       help='Optional metric to calculate')
    parser.add_argument('--sub-poiss', action='store_true',
                       help='Show only sub-Poissonian statistics (R ≥ 0) in plots')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for metric results')
    parser.add_argument('--num-analysis', action='store_true',
                       help='Enable numerical analysis of calculations')

    # Output options
    dir_group = parser.add_mutually_exclusive_group()
    dir_group.add_argument('--output-dir', type=Path,
                       default=Path('results'),
                       help='Output directory for new simulation results')
    dir_group.add_argument('--data-dir', type=Path,
                       help='Directory containing existing data to load (and save results to)')

    # Debug options
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    return parser.parse_args()

def setup_simulation(args: argparse.Namespace) -> SimulationConfig:
    """Create simulation configuration and setup output directories.

    Args:
        args: Parsed command line arguments

    Returns:
        Simulation configuration object
    """
    # Convert args to config dict
    config_dict = vars(args)
    config_dict['N_a'] = config_dict.pop('Na')
    config_dict['N_b'] = config_dict.pop('Nb')

    config_dict['use_gt_scale'] = config_dict.pop('gt_scale', False)
    if 'nu_t_scale' in config_dict:  # Remove if present since it's just a flag
        config_dict.pop('nu_t_scale')

    # Store and remove output_dir and data_dir from config_dict if present
    custom_output_dir = config_dict.pop('output_dir', None)
    data_dir = config_dict.pop('data_dir', None)

    # Create initial config
    config = SimulationConfig(**config_dict)

    # Get project root (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent

    # Set data_dir if provided (this takes precedence over output_dir)
    if data_dir:
        # If an absolute path is provided, use it as is
        if Path(data_dir).is_absolute():
            config.data_dir = Path(data_dir)
        else:
            # If relative path, make it relative to project root
            config.data_dir = project_root / data_dir

        # When data_dir is provided, we use it directly as the output directory
        # without creating a new simulation directory
        config.output_dir = config.data_dir

        # Create any missing subdirectories that might be needed
        for subdir in ['states', 'metrics', 'logs']:
            subdir_path = config.data_dir / subdir
            subdir_path.mkdir(exist_ok=True, parents=True)

        # Setup logging directly in the data directory
        log_file = config.data_dir / 'logs' / 'simulation.log'
        setup_logging(log_file, config.debug)

        # Log configuration details
        logger = logging.getLogger(__name__)
        logger.info(f"Using existing data directory: {config.data_dir}")

        # Return early since we don't need to create new directories
        return config

    # If no data_dir provided, create a new simulation directory
    if custom_output_dir:
        # If an absolute path is provided, use it as is
        if Path(custom_output_dir).is_absolute():
            base_output_dir = Path(custom_output_dir)
        else:
            # If relative path, make it relative to project root
            base_output_dir = project_root / custom_output_dir
    else:
        # Default to 'results' directory if neither is provided
        base_output_dir = project_root / 'results'

    # Generate simulation directory name
    sim_dirname = generate_simulation_dirname(config)

    # Create output directories
    output_dirs = create_output_dirs(
        base_output_dir / sim_dirname,
        ['states', 'metrics', 'logs']
    )

    # Update config with simulation-specific output directory
    config.output_dir = output_dirs['base']

    # Setup logging
    setup_logging(output_dirs['logs'] / 'simulation.log', config.debug)

    # Log configuration details with improved formatting
    logger = logging.getLogger(__name__)
    logger.info(f"Created output directories at: {output_dirs['base']}")
    logger.info("Simulation configuration:")
    logger.info(f"  - Hilbert space dimensions: Na={config.N_a}, Nb={config.N_b}")
    logger.info(f"  - Hamiltonian type: {config.hamiltonian}")
    logger.info(f"  - Initial state: {config.initial_state}")
    logger.info(f"  - Physical parameters:")
    logger.info(f"    · Trap frequency (ν): {config.nu}")
    logger.info(f"    · Lamb-Dicke (η): {config.eta}")
    logger.info(f"    · Coupling values (g/ν): {config.g_values}")

    # Log loss parameters if present
    if config.kappa is not None or config.gamma is not None or config.gamma_sp is not None:
        logger.info("  - Loss parameters:")
        if config.kappa is not None:
            logger.info(f"    · Cavity decay (κ): {config.kappa}")
        if config.gamma is not None:
            logger.info(f"    · Qubit dephasing (γ): {config.gamma}")
        if config.gamma_sp is not None:
            logger.info(f"    · Spontaneous emission (Γ): {config.gamma_sp}")

    logger.info("  - Evolution parameters:")
    logger.info(f"    · Time scale: {'gt' if config.use_gt_scale else 'νt'}")
    logger.info(f"    · Maximum time: {config.t_max}")
    logger.info(f"    · Time steps: {config.n_steps}")

    # Log state-specific parameters
    state_params = config.get_state_params()
    if state_params:
        logger.info("  - State parameters:")
        for key, value in state_params.items():
            logger.info(f"    · {key}: {value}")

    # Log additional options
    logger.info("  - Additional options:")
    logger.info(f"    · RWA: {config.rwa}")
    if config.metric:
        logger.info(f"    · Metric: {config.metric}")
        if config.plot:
            logger.info("    · Plot generation enabled")
    logger.info(f"    · Output directory: {config.output_dir}")
    if config.data_dir:
        logger.info(f"    · Data directory: {config.data_dir}")
    if config.profile or config.debug:
        logger.info("  - Debug options:")
        if config.profile:
            logger.info("    · Profiling enabled")
        if config.debug:
            logger.info("    · Debug output enabled")

    return config

def run_simulation(config: SimulationConfig) -> None:
    """Run quantum simulation with given configuration.

    Args:
        config: Simulation configuration
    """
    try:
        simulator = QuantumSimulator(config)
        if config.data_dir:
            logger.info(f"Using data directory for all operations: {config.data_dir}")
            config.output_dir = config.data_dir
            if config.metric:
                simulator._setup_metrics()
                metric_filepath = simulator._get_metric_results_filepath()
                if metric_filepath.exists():
                    logger.info(f"Found existing metric results file: {metric_filepath}")
                    if config.plot:
                        simulator.plot_metrics_from_file(metric_filepath)
                    return

                if simulator.metric_calculator:
                    logger.info("Calculating metrics from existing states...")
                    simulator.calculate_metrics_from_states(config.g_values)
                    if config.plot:
                        simulator.plot_metrics_from_file(metric_filepath)
                    return

        logger.info("Proceeding with full simulation run...")
        if simulator.simulate():
            save_simulation_data(simulator)

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

    logger.info("Workflow finished.")


def save_simulation_data(simulator: QuantumSimulator) -> None:
    """Save simulation data to files.

    Args:
        simulator: QuantumSimulator instance
    """
    output_dir = simulator.config.output_dir
    base_filename = simulator._get_base_filename()
    rwa_suffix = "_rwa" if simulator.config.rwa else "_full"

    # Save states and times
    states_dir = output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    for g, (times, states) in simulator.results.items():
        np.save(states_dir / f"{base_filename}{rwa_suffix}_times_g{g}.npy", times)
        qsave(states, states_dir / f"{base_filename}{rwa_suffix}_states_g{g}")

    # Save metrics
    if simulator.metric_results:
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metric_filepath = simulator._get_metric_results_filepath()
        save_dict = {}
        for g, (times, values) in simulator.metric_results.items():
            save_dict[f"g_{g}_times"] = times
            save_dict[f"g_{g}_values"] = values
        np.savez(metric_filepath, **save_dict)


def main() -> None:
    """Main entry point for the quantum simulation program."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Setup simulation configuration and directories
        config = setup_simulation(args)

        if config.profile:
            # Run with profiling
            profiler = cProfile.Profile()
            with Timer("Full simulation"):
                profiler.enable()
                run_simulation(config)
                profiler.disable()

            # Save and print profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats(config.output_dir / 'profile.stats')
            stats.print_stats(50)
        else:
            # Run normally
            with Timer("Full simulation"):
                run_simulation(config)

        logger.info("Simulation completed successfully!")

    except QuantumSimError as e:
        logger.error(f"Simulation error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()