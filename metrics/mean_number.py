"""
Mean Number Metric Module
=======================

This module provides the MeanNumberMetric class for calculating mean occupation
numbers in all system components.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, expect

from metrics.base import MetricCalculator
from config import MetricsConfig
from exceptions import MetricCalculationError, error_handler, StatePreparationError
from validators import StateValidator

# Setup logging
logger = logging.getLogger(__name__)

class MeanNumberMetric(MetricCalculator):
    """Calculator for mean occupation numbers in all system components."""

    def __init__(self):
        """Initialize mean number calculator."""
        super().__init__()
        self.components = ['Qubit', 'Cavity', 'Vibrational']
        self.plot_types = ['sigma_z', 'excitation_prob']

    @error_handler(MetricCalculationError)
    def calculate(self, states: List[Qobj], times: np.ndarray, g_value: float) -> np.ndarray:
        """Calculate mean occupation numbers for each component.

        Calculates:
            - Qubit expectation value <σz> (ranges from -1 to 1)
            - Mean cavity photon number <b†b>
            - Mean vibrational phonon number <a†a>

        Note: <σz> is NOT the qubit excitation probability directly. It represents
        the difference between excited and ground state probabilities: <σz> = Pe - Pg.
        The qubit excitation probability Pe can be calculated as Pe = (<σz> + 1) / 2.

        Args:
            states: States from quantum evolution (either Full or RWA)
            times: Array of time points
            g_value: The coupling strength value

        Returns:
            Array of shape (n_times, 3) containing:
                [:, 0] - Qubit <σz>
                [:, 1] - Cavity <n_b>
                [:, 2] - Vibrational <n_a>
        """
        self.log_progress("Starting mean occupation numbers calculation")

        try:
            if not states:
                raise ValueError("States must be provided")

            # Validate states
            validation = StateValidator.validate_state_evolution(times, states)
            if not validation.is_valid:
                raise StatePreparationError(validation.message)

            # Initialize operators if needed
            if not self.operators:
                dims = states[0].dims[0]
                self.initialize_operators(dims[2], dims[1])

            # Initialize numerical analysis for this g-value
            if self._num_analysis_enabled:
                self._analyzer.start_g_value_analysis(g_value)

            # Get required operators
            operators = {
                'qubit': self.operators.get_operator('sigma_z'),  # σz for qubit excitation
                'cavity': self.operators.get_operator('b_op').dag() * self.operators.get_operator('b_op'),  # b†b
                'vib': self.operators.get_operator('a_op').dag() * self.operators.get_operator('a_op')  # a†a
            }

            mean_values = []
            total_states = len(states)

            for idx, state in enumerate(states):
                values = []

                # Process evolution values
                for op_name, operator in operators.items():
                    if self._num_analysis_enabled:
                        # Analyze expectation values
                        diag = self._analyzer.analyze_expectation(
                            state, operator, f'step_{idx}_{op_name}')
                        values.append(diag.values['value_real'])
                    else:
                        values.append(expect(operator, state).real)

                mean_values.append(values)

                if idx % (total_states // 10) == 0:
                    self.log_progress(f"Processed {idx+1}/{total_states} states")

            # Save diagnostics if enabled
            if self._num_analysis_enabled:
                self.save_diagnostics()

            return np.array(mean_values)

        except Exception as e:
            self.log_progress(f"Mean numbers calculation failed: {str(e)}", 'error')
            raise MetricCalculationError("Failed to calculate mean numbers") from e

    def compute_qubit_excitation_probability(self, sigma_z_values: np.ndarray) -> np.ndarray:
        """Compute qubit excitation probability from sigma_z expectation values.

        The qubit excitation probability Pe is related to <σz> by: Pe = (<σz> + 1) / 2

        Args:
            sigma_z_values: Array of <σz> expectation values

        Returns:
            Array of qubit excitation probabilities
        """
        return (sigma_z_values + 1) / 2


    @error_handler(MetricCalculationError)
    def plot(self, base_filename: str,
            values: Dict[float, Tuple[np.ndarray, np.ndarray]],
            config: MetricsConfig, title_suffix: str = '') -> None:
        """Plot mean occupation numbers evolution.

        Creates separate plots for:
        1. <σz> values (ranges from -1 to 1)
        2. Qubit excitation probability Pe (ranges from 0 to 1)
        3. Mean cavity photon number
        4. Mean vibrational phonon number

        Each plot shows the evolution for a single simulation run (either Full or RWA).
        """
        if not config.plot:
            return

        # Get g-values
        g_values = sorted(values.keys())

        # Log data structure for debugging
        for g_value in g_values:
            scaled_t, metric_vals = values[g_value]
            self.log_progress(f"Data for g={g_value}: times shape={scaled_t.shape}, values shape={metric_vals.shape}", level='debug')

        # Create plots for each type (sigma_z and excitation_prob)
        for plot_type in self.plot_types:
            # Determine plot settings based on type
            if plot_type == 'sigma_z':
                # For sigma_z, we only plot the qubit component
                components_to_plot = ['Qubit']
                title = 'Qubit <σz> Evolution'
                suffix = '_sigma_z'
            else:  # excitation_prob
                # For excitation probability, we plot all components
                components_to_plot = self.components
                title = 'Mean Occupation Numbers Evolution'
                suffix = '_mean_numbers'

            # Create a figure for each g-value
            for g_value in g_values:
                # Get data for this g-value
                scaled_t, metric_vals = values[g_value]

                # Check if metric_vals is 1D or 2D
                is_1d = len(metric_vals.shape) == 1

                # Log the shape for debugging
                self.log_progress(f"Metric values shape for g={g_value}: {metric_vals.shape}", level='debug')

                # If 1D, reshape to 2D with a single column
                if is_1d:
                    self.log_progress(f"Detected 1D metric values, reshaping to 2D", level='debug')
                    # Reshape to (n_times, 1)
                    metric_vals = metric_vals.reshape(-1, 1)
                    # Only plot the first component (Qubit) for 1D data
                    if plot_type == 'excitation_prob':
                        self.log_progress(f"Only plotting Qubit component for 1D data", level='info')
                        components_to_plot = ['Qubit']
                elif metric_vals.shape[1] < len(self.components):
                    # If we have fewer columns than components, adjust components_to_plot
                    self.log_progress(f"Data has {metric_vals.shape[1]} columns but {len(self.components)} components", level='info')
                    self.log_progress(f"Only plotting available components", level='info')
                    components_to_plot = self.components[:metric_vals.shape[1]]

                # Create figure with subplots for each component
                fig, axes = plt.subplots(len(components_to_plot), 1,
                                        figsize=(8, 3 * len(components_to_plot)),
                                        sharex=True)

                # Convert to array of axes if there's only one subplot
                if len(components_to_plot) == 1:
                    axes = [axes]

                # Plot each component
                for i, component in enumerate(components_to_plot):
                    ax = axes[i]
                    comp_idx = min(self.components.index(component), metric_vals.shape[1]-1)

                    # Get the data to plot based on plot type and component
                    if plot_type == 'sigma_z' or component != 'Qubit':
                        # For sigma_z plot or non-qubit components, use the raw values
                        y_values = metric_vals[:, comp_idx]
                    else:  # excitation_prob for Qubit
                        # For qubit excitation probability, convert sigma_z to Pe
                        y_values = self.compute_qubit_excitation_probability(metric_vals[:, comp_idx])

                    # Plot the data
                    ax.plot(scaled_t, y_values, color='black')

                    # Set labels
                    if plot_type == 'sigma_z' and component == 'Qubit':
                        ax.set_ylabel('Qubit <σz>')
                    elif plot_type == 'excitation_prob' and component == 'Qubit':
                        ax.set_ylabel('Qubit Excitation\nProbability (Pe)')
                    else:
                        ax.set_ylabel(f'Mean {component}\nOccupation')

                    # Add component-specific features
                    if component == 'Qubit':
                        if plot_type == 'sigma_z':
                            # For sigma_z, show the full range from -1 to 1
                            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
                            ax.set_ylim(-1.1, 1.1)
                        else:  # excitation_prob
                            # For Pe, show the range from 0 to 1
                            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
                            ax.set_ylim(-0.1, 1.1)

                    # Add grid
                    ax.grid(config.plot_style.grid)

                # Set common x-label
                time_label = 'gt' if config.use_gt_scale else 'νt'
                axes[-1].set_xlabel(time_label)

                # Set title
                plt.suptitle(f'{title} for g/ν = {g_value}{title_suffix}')
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout but leave room for suptitle

                # Save figure
                output_path = config.output_dir / f'{base_filename}{suffix}_g{g_value}{title_suffix}.png'
                plt.savefig(output_path, dpi=config.plot_style.dpi, bbox_inches='tight')
                plt.close()
