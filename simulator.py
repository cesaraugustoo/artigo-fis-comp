"""
Quantum System Simulator Module
=============================

This module provides the core simulation engine for the trapped ion-cavity quantum
system. It integrates state preparation, Hamiltonian construction, and time
evolution calculations.

Classes:
    MemoryMonitor: Tracks memory usage during simulation
    QuantumSimulator: Main simulator class
"""

import gc
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from qutip import mesolve, Qobj, qsave
import psutil

from config import SimulationConfig, MetricsConfig
from exceptions import (
    SimulationError, TimeEvolutionError,
    error_handler
)
from operators import QuantumOperators
from dissipation import CollapseOperatorBuilder
from states import StateGenerator
from hamiltonians import HamiltonianBuilder
from utils import clear_matplotlib
from validators import StateValidator

# Setup logging
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage during simulation."""

    def __init__(self, debug: bool = False):
        """Initialize memory monitor.

        Args:
            debug: Enable detailed memory logging
        """
        self.process = psutil.Process()
        self.debug = debug
        self.start_memory = self.current_memory()
        self._snapshots = {}

    def current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def log_memory(self, operation: str):
        """Log memory usage for an operation."""
        if self.debug:
            current = self.current_memory()
            diff = current - self.start_memory
            logger.debug(f"Memory {operation}: {current:.1f} MB (Δ: {diff:+.1f} MB)")
            self._snapshots[operation] = current

    def get_peak_usage(self) -> Tuple[str, float]:
        """Get operation with highest memory usage."""
        if not self._snapshots:
            return None, 0
        peak_op = max(self._snapshots.items(), key=lambda x: x[1])
        return peak_op[0], peak_op[1]

class QuantumSimulator:
    """Main simulator class for quantum system evolution."""

    def __init__(self, config: SimulationConfig):
        """Initialize quantum simulator.

        Args:
            config: Simulation configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate configuration
        config.validate()
        self.config = config

        # Initialize memory monitor
        self.memory = MemoryMonitor(config.debug)

        # Initialize components
        self.operators = QuantumOperators(config.N_a, config.N_b)
        self.state_generator = StateGenerator(config.N_a, config.N_b)
        self.hamiltonian_builder = HamiltonianBuilder(
            config.N_a, config.N_b, config.nu, config.eta)

        # Initialize collapse operator builder
        self.collapse_builder = CollapseOperatorBuilder(config.N_a, config.N_b)

        # Initialize metric calculator if specified
        self.metric_calculator = None
        self.metrics_config = None
        if config.metric:
            self._setup_metrics()

        self.results = {}
        self.metric_results = {}

        self.memory.log_memory("after initialization")

    def _setup_metrics(self):
        """Setup metrics configuration and calculator."""
        try:
            # Create metrics directory inside the base output directory
            metrics_dir = self.config.output_dir / 'metrics' / self.config.metric
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Initialize metrics configuration
            self.metrics_config = MetricsConfig(
                metric_type=self.config.metric,
                output_dir=metrics_dir,
                plot=self.config.plot,
                use_gt_scale=self.config.use_gt_scale,
                sub_poiss_only=getattr(self.config, 'sub_poiss', False),
                num_analysis=self.config.num_analysis
            )

            # Get appropriate metric calculator from registry
            from metrics import get_calculator
            self.metric_calculator = get_calculator(self.config.metric)

            # Setup numerical analysis if enabled
            if self.metrics_config.num_analysis:
                self.metric_calculator.setup_numerical_analysis(
                    enabled=True,
                    output_dir=metrics_dir
                )

        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")
            raise SimulationError("Metrics setup failed") from e

    @error_handler(TimeEvolutionError)
    def _run_evolution(self, initial_state: Qobj, g_value: float
                      ) -> Tuple[np.ndarray, List[Qobj]]:
        """Run quantum evolution for a specific coupling strength.

        Args:
            initial_state: Initial quantum state
            g_value: Coupling strength value

        Returns:
            Tuple of (times array, evolved states list)
        """
        try:
            # Get time points
            times, scaled_times = self.config.get_evolution_times(g_value)

            # Get Hamiltonian
            hamiltonian = self.hamiltonian_builder.build_hamiltonian(
                self.config.hamiltonian, g_value, self.config.rwa)

            # Build collapse operators if loss parameters are provided
            c_ops = self.collapse_builder.build_collapse_operators(
                kappa=self.config.kappa,
                gamma=self.config.gamma,
                gamma_sp=self.config.gamma_sp
            )

            # Log dissipation information
            if c_ops:
                logger.info(f"Running evolution with {len(c_ops)} collapse operators:")
                if self.config.kappa is not None and self.config.kappa > 0:
                    logger.info(f"  - Cavity decay rate κ = {self.config.kappa}")
                if self.config.gamma is not None and self.config.gamma > 0:
                    logger.info(f"  - Qubit dephasing rate γ = {self.config.gamma}")
                if self.config.gamma_sp is not None and self.config.gamma_sp > 0:
                    logger.info(f"  - Spontaneous decay rate Γ = {self.config.gamma_sp}")
            else:
                logger.info("Running ideal evolution (no losses)")

            # Run evolution with collapse operators if present
            result = mesolve(hamiltonian, initial_state, times, c_ops,
                           options=self.config.solver_options)

            # Validate evolution results
            validation = StateValidator.validate_state_evolution(
                scaled_times, result.states)

            if not validation.is_valid:
                raise SimulationError(
                    f"Evolution validation failed: {validation.message}")

            return scaled_times, result.states

        except Exception as e:
            logger.error(f"Evolution failed for g={g_value}: {str(e)}")
            raise TimeEvolutionError(f"Evolution failed: {str(e)}")

    def _run_simulation(self, initial_state: Qobj) -> Dict:
        """Run quantum simulation.

        Args:
            initial_state: Initial quantum state

        Returns:
            Dictionary of results
        """
        base_filename = self._get_base_filename()

        try:
            for g_value in self.config.g_values:
                logger.info(f"Simulating for g/nu = {g_value}")

                # Run evolution
                scaled_times, states = self._run_evolution(initial_state, g_value)

                # Always save states regardless of metric calculation
                self._save_states(states, scaled_times, g_value, base_filename)

                self.results[g_value] = (scaled_times, states)

                if self.metric_calculator:
                    # Calculate metric values - pass g_value to the calculator
                    metric_values = self.metric_calculator.calculate(
                        states, scaled_times, g_value)
                    self.metric_results[g_value] = (scaled_times, metric_values)

                # Clear intermediate results
                del states
                gc.collect()
                clear_matplotlib()

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

        return self.results

    def _get_base_filename(self) -> str:
        """Generate base filename for outputs."""
        return (f'{self.config.N_a}_{self.config.hamiltonian}_'
                f'{self.config.initial_state}')

    def _get_metric_results_filepath(self) -> Path:
        """Get the path to the metric results file."""
        if not self.metrics_config or not self.config.metric:
            raise ValueError("Metrics configuration not set up or no metric specified")

        base_filename = self._get_base_filename()

        # Add the _full or _rwa suffix to the filename
        suffix = '_rwa' if self.config.rwa else '_full'
        if '_full' not in base_filename and '_rwa' not in base_filename:
            base_filename = f'{base_filename}{suffix}'

        output_filename = f'{base_filename}_{self.config.metric}_values.npz'

        # Ensure metrics directory exists
        metrics_dir = self.config.output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)

        # Create metric-specific directory
        metric_dir = metrics_dir / self.config.metric
        metric_dir.mkdir(exist_ok=True)

        # Update metrics_config to use the correct output directory
        if hasattr(self, 'metrics_config'):
            self.metrics_config.output_dir = metric_dir

        return metric_dir / output_filename

    def _get_state_times_filepaths(self, g_value: float, rwa_flag: bool = None) -> tuple[Path, Path]:
        """Get the paths to the state and time files for a given g_value."""
        # Ensure states directory exists
        states_dir = self.config.output_dir / 'states'

        # Generate filenames
        base_filename = self._get_base_filename()
        logger.debug(f"Base filename: {base_filename}")

        # Determine if we're looking for RWA or full files
        use_rwa = self.config.rwa if rwa_flag is None else rwa_flag
        suffix = '_rwa' if use_rwa else '_full'
        prefix = f"{base_filename}{suffix}"
        logger.debug(f"Prefix: {prefix}")

        states_filename = f'{prefix}_states_g{g_value}'
        times_filename = f'{prefix}_times_g{g_value}.npy'

        logger.debug(f"Generated state filename: {states_filename}")
        logger.debug(f"Generated times filename: {times_filename}")

        return (states_dir / states_filename, states_dir / times_filename)

    def _save_states(self, states: List[Qobj], times: np.ndarray,
                    g_value: float, base_filename: str) -> None:
        """Save quantum states and times to files."""
        # Get file paths
        states_dir = self.config.output_dir / 'states'
        states_dir.mkdir(exist_ok=True)

        # Append _full or _rwa to the base_filename based on the RWA setting
        modified_base_filename = base_filename
        if '_full' not in base_filename and '_rwa' not in base_filename:
            suffix = '_rwa' if self.config.rwa else '_full'
            modified_base_filename = f'{base_filename}{suffix}'

        # Generate filenames
        states_filename = f'{modified_base_filename}_states_g{g_value}'
        times_filename = f'{modified_base_filename}_times_g{g_value}.npy'

        # Save files
        qsave(states, states_dir / states_filename)
        np.save(states_dir / times_filename, times)

        logger.info(f"Saved evolution data for g={g_value}:")
        logger.info(f"  - States: {states_filename}.qu")
        logger.info(f"  - Times: {times_filename}")

    def simulate(self) -> bool:
        """Run quantum simulation according to configuration."""
        self.memory.log_memory("before simulation")
        start_time = time.time()

        try:
            # Save the current configuration to YAML
            try:
                config_save_path = self.config.output_dir / 'simulation_config.yaml'
                self.config.to_yaml(config_save_path)
                logger.info(f"Saved simulation configuration to {config_save_path}")
            except Exception as e:
                logger.warning(f"Could not save simulation configuration: {e}")

            # Generate initial state with parameters from config
            logger.info("Generating initial state")
            initial_state = self.state_generator.generate_initial_state(
                self.config.initial_state,
                self.config.get_state_params()
            )

            # Run simulation
            logger.info("Running simulation")
            self.results = self._run_simulation(initial_state)

            # Calculate and save metrics if requested
            if self.config.metric and self.metric_calculator:
                logger.info(f"Calculating {self.config.metric} metrics from simulation results")
                self._save_results(self.metric_results)

                # Generate plots if requested
                if self.config.plot:
                    logger.info("Generating plots from calculated metrics")
                    # Use the results we already have instead of reloading from file
                    base_filename = self._get_base_filename()
                    ham_suffix = ' for H+' if self.config.hamiltonian == 'h_plus' else ' for H-'
                    logger.info("Generating metric plots directly from calculated results")
                    self.metric_calculator.plot(base_filename, self.metric_results, self.metrics_config, ham_suffix)

            elapsed = time.time() - start_time
            logger.info(f"Simulation completed in {elapsed:.2f} seconds")

            # Log memory statistics
            peak_op, peak_mem = self.memory.get_peak_usage()
            if peak_op:
                logger.info(f"Peak memory usage: {peak_mem:.1f} MB during {peak_op}")

            return True

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            return False

        finally:
            self.cleanup()

    def _save_results(self, results: Dict):
            """Save simulation results."""
            base_filename = self._get_base_filename()
            ham_suffix = ' for H+' if self.config.hamiltonian == 'h_plus' else ' for H-'

            logger.info(f"Saving {self.config.metric} metric results")

            # Ensure metrics directory exists
            metrics_dir = self.config.output_dir / 'metrics'
            metrics_dir.mkdir(exist_ok=True)

            # Create metric-specific directory
            metric_dir = metrics_dir / self.config.metric
            metric_dir.mkdir(exist_ok=True)

            # Update metrics_config to use the correct output directory
            self.metrics_config.output_dir = metric_dir

            if self.config.plot:
                logger.info("Generating metric plots")
                self.metric_calculator.plot(base_filename, results,
                                        self.metrics_config, ham_suffix)

            # Save numerical results
            # Modify the output path to include _full or _rwa
            suffix = '_rwa' if self.config.rwa else '_full'
            base_filename_with_suffix = f'{base_filename}{suffix}'
            output_filename = f'{base_filename_with_suffix}_{self.config.metric}_values.npz'
            output_path = metric_dir / output_filename

            logger.info(f"Saving metric data to: {output_path}")

            # Create a dictionary of arrays to save
            save_dict = {}
            for g, (times, values) in results.items():
                save_dict[f"g_{g}_times"] = times
                save_dict[f"g_{g}_values"] = values

            # Save the data
            np.savez(output_path, **save_dict)
            logger.info(f"Saved metric data to: {output_path}")

    def plot_metrics_from_file(self, metric_results_filepath: Path) -> None:
        """Plot metrics from a previously saved results file."""
        if not metric_results_filepath.exists():
            raise FileNotFoundError(f"Metric results file not found: {metric_results_filepath}")

        # Ensure metric calculator is set up
        if not self.metric_calculator:
            self._setup_metrics()

        try:
            # Load results
            logger.info(f"Loading metric results from {metric_results_filepath}")
            data = np.load(metric_results_filepath)

            # Print all keys in the file for debugging
            logger.info(f"Keys in the .npz file: {data.files}")

            # Reconstruct results dictionary
            results = {}
            g_values = sorted(list(set([key.split('_')[1] for key in data.files])))
            for g_value in g_values:
                g_value_float = float(g_value)
                times = data[f"g_{g_value}_times"]
                values = data[f"g_{g_value}_values"]
                results[g_value_float] = (times, values)

            # Plot results
            base_filename = self._get_base_filename()
            ham_suffix = ' for H+' if self.config.hamiltonian == 'h_plus' else ' for H-'

            # Ensure metrics directory exists in the output directory
            metrics_dir = self.config.output_dir / 'metrics'
            metrics_dir.mkdir(exist_ok=True)

            # Create metric-specific directory
            metric_dir = metrics_dir / self.config.metric
            metric_dir.mkdir(exist_ok=True)

            # Update metrics_config to use the correct output directory
            self.metrics_config.output_dir = metric_dir

            logger.info(f"Generating metric plots in {metric_dir}")
        except Exception as e:
            logger.error(f"Error loading or processing metric data: {str(e)}")
            raise

        # Check if this is a metric that requires states for additional plots
        if self.config.metric in ['r_param', 'wigner_neg']:
            # Try to find the state files for the g-values in the results
            g_values = set()
            for key in results.keys():
                if isinstance(key, tuple) and len(key) == 2:
                    g_values.add(key[0])
                else:
                    g_values.add(key)

            # Check if we can find all the state files
            all_states_found = True
            states_by_g = {}

            for g_value in g_values:
                try:
                    # Get paths for states - use the same RWA flag as the metric file
                    rwa_flag = '_rwa' in str(metric_results_filepath)
                    states_path, times_path = self._get_state_times_filepaths(g_value, rwa_flag)

                    # Log the paths we're looking for
                    logger.info(f"Looking for state files for g={g_value}:")
                    logger.info(f"  - States path: {states_path}")
                    logger.info(f"  - Times path: {times_path}")

                    # Check if we should use data_dir
                    if self.config.data_dir:
                        # Check if data_dir already has a 'states' subdirectory
                        if (self.config.data_dir / 'states').exists():
                            states_path = self.config.data_dir / 'states' / states_path.name
                            times_path = self.config.data_dir / 'states' / times_path.name
                        else:
                            # Try without the 'states' subdirectory
                            states_path = self.config.data_dir / states_path.name
                            times_path = self.config.data_dir / times_path.name

                        logger.info(f"Using data_dir, looking in:")
                        logger.info(f"  - States path: {states_path}")
                        logger.info(f"  - Times path: {times_path}")

                        # List all files in the directory to help with debugging
                        try:
                            import os
                            parent_dir = states_path.parent
                            logger.info(f"Files in {parent_dir}:")
                            if parent_dir.exists():
                                for file in os.listdir(parent_dir):
                                    logger.info(f"  - {file}")
                            else:
                                logger.info(f"  Directory does not exist")
                        except Exception as e:
                            logger.warning(f"Error listing files: {str(e)}")

                    # Check if files exist (state files have .qu extension added by QuTiP)
                    # Try both with and without the .qu extension
                    states_file = states_path.with_suffix('.qu')
                    states_exists = states_file.exists() or states_path.exists()
                    times_exists = times_path.exists()

                    logger.info(f"File existence check:")
                    logger.info(f"  - States file with .qu: {states_file.exists()}")
                    logger.info(f"  - States file without .qu: {states_path.exists()}")
                    logger.info(f"  - Times file: {times_exists}")

                    # If files not found, try looking for files with a different pattern
                    if not states_exists or not times_exists:
                        logger.info("Files not found with standard pattern, trying alternative patterns...")
                        try:
                            import glob
                            # Look for any state file with the g-value in the name
                            parent_dir = states_path.parent
                            if parent_dir.exists():
                                # Pattern for state files
                                state_pattern = f"*_states_g{g_value}*.qu"
                                time_pattern = f"*_times_g{g_value}*.npy"

                                state_matches = list(parent_dir.glob(state_pattern))
                                time_matches = list(parent_dir.glob(time_pattern))

                                logger.info(f"Found {len(state_matches)} state files matching pattern {state_pattern}")
                                for match in state_matches:
                                    logger.info(f"  - {match.name}")

                                logger.info(f"Found {len(time_matches)} time files matching pattern {time_pattern}")
                                for match in time_matches:
                                    logger.info(f"  - {match.name}")

                                if state_matches and time_matches:
                                    states_path = state_matches[0]
                                    times_path = time_matches[0]
                                    states_exists = True
                                    times_exists = True
                                    logger.info(f"Using alternative files:")
                                    logger.info(f"  - States: {states_path}")
                                    logger.info(f"  - Times: {times_path}")
                        except Exception as e:
                            logger.warning(f"Error searching for alternative files: {str(e)}")

                    if states_exists and times_exists:
                        # Load states and times
                        from qutip import qload
                        logger.info(f"Found state files for g={g_value}, loading for additional plots")

                        # Remove .qu extension if present to avoid double extension
                        states_path_clean = str(states_path)
                        if states_path_clean.endswith('.qu'):
                            states_path_clean = states_path_clean[:-3]

                        states = qload(states_path_clean)
                        times = np.load(times_path)

                        # Store for later use
                        states_by_g[g_value] = (states, times)
                    else:
                        logger.warning(f"Could not find state files for g={g_value}")
                        all_states_found = False
                except Exception as e:
                    logger.warning(f"Error loading state files for g={g_value}: {str(e)}")
                    all_states_found = False

            if not all_states_found:
                logger.warning(f"Note: When loading {self.config.metric} data from file without all state files,")
                logger.warning("only basic plots will be generated.")
                logger.warning("Additional plots (like photon distributions or Wigner functions) require the original states.")
            else:
                # Pass the loaded states to the metric calculator
                logger.info("Found all state files, will generate additional plots")
                self.metric_calculator.set_states(states_by_g)

        try:
            # Check if we have any results to plot
            if not results:
                logger.warning("No valid data found to plot")
                return

            # Log the results we're plotting
            logger.info(f"Plotting results for {len(results)} g-values: {sorted(results.keys())}")

            # For each g-value, log the shape of the data
            for g_value, (times, values) in results.items():
                shape_str = 'scalar'
                if hasattr(values, 'shape'):
                    shape_str = f"{values.shape}"
                    if len(values.shape) > 1 and values.shape[1] > 1:
                        logger.info(f"Multi-column data detected for g={g_value} with {values.shape[1]} columns")
                logger.info(f"Data for g={g_value}: times shape={times.shape}, values shape={shape_str}")

            # Plot the results
            self.metric_calculator.plot(base_filename, results, self.metrics_config, ham_suffix)
            logger.info(f"Successfully plotted metrics")
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources after simulation."""
        self.operators.clear_cache()
        self.state_generator.cleanup()
        self.hamiltonian_builder.cleanup()
        self.collapse_builder.cleanup()
        if hasattr(self, 'metric_calculator') and self.metric_calculator:
            self.metric_calculator.clear_cache()
        gc.collect()
        clear_matplotlib()