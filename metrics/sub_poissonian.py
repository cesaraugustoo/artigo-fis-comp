"""
Sub-Poissonian Metric Module
==========================

This module provides the SubPoissonianMetric class for calculating sub-Poissonian
statistics using parameter R and plotting cavity photon number distributions.
"""

import logging
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, expect

# Assuming Path is needed for output directories defined in config
from pathlib import Path

from metrics.base import MetricCalculator
from config import MetricsConfig
from exceptions import MetricCalculationError, error_handler
from validators import StateValidator, StatePreparationError
from numerical_analysis import DiagnosticResult

# Setup logging
logger = logging.getLogger(__name__)

class SubPoissonianMetric(MetricCalculator):
    """Calculator for sub-Poissonian statistics using parameter R.

    The R parameter quantifies the deviation from Poissonian statistics:
        R = 1 - ⟨(Δn)²⟩/⟨n⟩
    where ⟨(Δn)²⟩ is the cavity photon number variance and ⟨n⟩ is the mean
    cavity photon number.

    R > 0: Sub-Poissonian statistics (quantum)
    R = 0: Poissonian statistics (coherent)
    R < 0: Super-Poissonian statistics (thermal)

    Also provides functionality to plot the cavity photon number distribution
    P(n) = ⟨n|ρ_cavity|n⟩ at times corresponding to minimum and maximum R values.
    """

    @error_handler(MetricCalculationError)
    def calculate(self, states: List[Qobj], times: np.ndarray, g_value: float) -> np.ndarray:
        """Calculate the R parameter for a sequence of states.

        Args:
            states: List of quantum states (full system: qubit @ cavity @ vibrational).
            times: Array of time points corresponding to the states.
            g_value: Coupling strength value (for diagnostics).

        Returns:
            Array of R parameter values for the cavity field.

        Raises:
            MetricCalculationError: If calculation fails.
        """
        self.log_progress("Starting R parameter calculation for sub-Poissonian statistics")

        try:
            # Cache states for potential plotting later
            self._cache['states'] = states
            self._cache['times'] = times # Cache times as well for plotting labels

            # Validate states
            validation = StateValidator.validate_state_lists(states, check_dims=True)
            if not validation.is_valid:
                raise StatePreparationError(validation.message)

            # Initialize operators if not already done
            if not self.operators:
                dims = states[0].dims[0]
                # Ensure dimensions are correct [qubit, cavity, vibrational]
                if len(dims) != 3:
                    raise ValueError(f"Expected 3 subsystems, but state dimensions are {dims}")
                N_a = dims[2] # Vibrational dimension
                N_b = dims[1] # Cavity dimension
                self.initialize_operators(N_a, N_b)

            # Get cavity operators defined on the full Hilbert space
            b_op = self.operators.get_operator('b_op') # Should be Iq @ b @ Ia
            if b_op is None:
                raise OperatorError("Cavity operator 'b_op' not found.")
            number_op = b_op.dag() * b_op
            # Create number squared operator (b†b)²
            number_squared_op = number_op * number_op

            # Initialize numerical analysis for this g-value
            if self._num_analysis_enabled:
                self._analyzer.start_g_value_analysis(g_value)

            r_values = []
            total_states = len(states)
            log_interval = max(1, total_states // 10) # Log progress roughly 10 times

            for idx, state in enumerate(states):
                # --- Expectation Value Calculation ---
                if self._num_analysis_enabled:
                    # Analyze input state
                    self._analyzer.analyze_state(state, f'step_{idx}_state')

                    # Calculate expectation values with error tracking
                    n_mean_diag = self._analyzer.analyze_expectation(
                        state, number_op, f'step_{idx}_mean')
                    n_squared_diag = self._analyzer.analyze_expectation(
                        state, number_squared_op, f'step_{idx}_squared')

                    # Check if analysis was successful before accessing values
                    if n_mean_diag and 'value_real' in n_mean_diag.values:
                        n_mean = n_mean_diag.values['value_real']
                    else:
                        self.log_progress(f"Warning: Could not get mean value from analyzer for state {idx}. Recalculating.", level='warning')
                        n_mean = expect(number_op, state).real

                    if n_squared_diag and 'value_real' in n_squared_diag.values:
                        n_squared = n_squared_diag.values['value_real']
                    else:
                        self.log_progress(f"Warning: Could not get squared mean value from analyzer for state {idx}. Recalculating.", level='warning')
                        n_squared = expect(number_squared_op, state).real

                    # Analyze variance
                    var_diag = self._analyzer.analyze_variance(
                        n_mean, n_squared, f'step_{idx}_variance')
                    variance = var_diag.values.get('variance', n_squared - n_mean**2) # Fallback calculation
                else:
                    # Direct calculation without error tracking
                    n_mean = expect(number_op, state).real
                    n_squared = expect(number_squared_op, state).real
                    variance = n_squared - n_mean**2
                # --- End Expectation Value Calculation ---

                # --- Calculate R parameter ---
                # Use a small tolerance for the mean to avoid division by near-zero
                if abs(n_mean) > 1e-10:
                    # Ensure variance is not negative due to numerical precision issues
                    variance = max(0.0, variance)
                    r_value = 1.0 - variance / n_mean
                else:
                    # If mean is essentially zero, variance should also be zero.
                    # Define R=0 (Poissonian) for the vacuum state.
                    r_value = 0.0
                # --- End R Parameter Calculation ---

                r_values.append(r_value)

                # Log progress periodically
                if (idx + 1) % log_interval == 0 or (idx + 1) == total_states:
                    self.log_progress(f"Processed {idx+1}/{total_states} states")

            # Save diagnostics if enabled
            if self._num_analysis_enabled:
                self.save_diagnostics()

            self.log_progress("R parameter calculation completed.")
            return np.array(r_values)

        except Exception as e:
            self.log_progress(f"R parameter calculation failed: {str(e)}", 'error')
            # Re-raise as MetricCalculationError for consistent error handling upstream
            raise MetricCalculationError(f"Failed to calculate R parameter: {e}") from e

    @error_handler(MetricCalculationError)
    def plot(self, base_filename: str,
            values: Dict[float, Tuple[np.ndarray, np.ndarray]],
            config: MetricsConfig, title_suffix: str = '') -> None:
        """Plot R parameter evolution and cavity photon distributions at key points.

        Args:
            base_filename: Base name for output files.
            values: Dictionary mapping coupling strengths (g_value) to
                    (time_array, r_value_array).
            config: Metrics configuration object.
            title_suffix: Optional suffix for plot titles and filenames.
        """
        if not getattr(config, 'plot', False): # Check if plotting is enabled
            self.log_progress("Plotting is disabled in config.")
            return

        if not values:
            self.log_progress("No data provided for plotting R evolution.", level='warning')
            return

        # --- Plot R Parameter Evolution ---
        try:
            plt.figure(figsize=getattr(config.plot_style, 'figsize', (10, 6))) # Use default figsize if not set
            num_g_values = len(values)
            # Use a perceptually uniform colormap if multiple lines
            colors = plt.cm.viridis(np.linspace(0, 1, num_g_values)) if num_g_values > 1 else ['#1f77b4'] # Default blue

            all_r_values_list = []
            plotted_g_values = [] # Keep track of g_values actually plotted
            for i, (g_value, (scaled_t, r_values)) in enumerate(values.items()):
                if len(scaled_t) != len(r_values):
                     self.log_progress(f"Warning: Mismatch in length of time ({len(scaled_t)}) and R ({len(r_values)}) arrays for g/ν={g_value}. Skipping plot for this value.", level='warning')
                     continue
                if len(r_values) == 0:
                    self.log_progress(f"Warning: Empty R values array for g/ν={g_value}. Skipping plot for this value.", level='warning')
                    continue

                plt.plot(scaled_t, r_values, label=f'g/ν={g_value}',
                        color=colors[i], linewidth=2)
                all_r_values_list.append(r_values)
                plotted_g_values.append(g_value)

            if not all_r_values_list: # Check if any data was actually plotted
                 self.log_progress("No valid data to plot for R evolution.", level='warning')
                 plt.close() # Close the empty figure
                 # Continue to potentially plot distributions if states are cached
            else:
                all_r_values = np.concatenate(all_r_values_list)
                # Determine y-limits based on config and data
                sub_poiss_only = getattr(config, 'sub_poiss_only', False) # Default to False if not set
                y_min_data = np.min(all_r_values)
                y_max_data = np.max(all_r_values)
                # Set lower limit, respecting sub_poiss_only flag
                y_min = 0 if sub_poiss_only and y_min_data >= -1e-9 else y_min_data # Allow slightly below 0
                # Add some margin, ensuring max is not less than min if data range is small
                y_max = y_max_data + max(0.1, abs(y_max_data) * 0.1)
                if y_max <= y_min + 1e-9: # Handle cases where data is constant or range is tiny
                    y_max = y_min + 0.1

                time_label = 'gt' if getattr(config, 'use_gt_scale', False) else 'νt'
                plt.xlabel(time_label)
                plt.ylabel('R Parameter')
                title = 'Sub-Poissonian Statistics Evolution'
                if sub_poiss_only:
                    title = 'Sub-Poissonian Statistics Evolution (R ≥ 0 only)'
                title += title_suffix
                plt.title(title)

                plt.grid(getattr(config.plot_style, 'grid', True), alpha=0.5) # Use getattr for safety

                # Add reference line for Poissonian statistics
                plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Poissonian (R=0)')
                plt.ylim(y_min, y_max)

                # Add legend if multiple curves or if the reference line label is desired
                if num_g_values > 1 or plt.gca().get_legend_handles_labels()[1]: # Check if any labels exist
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                output_path_r = config.output_dir / f'{base_filename}_subpoissonian{title_suffix}.png'
                plt.savefig(output_path_r,
                           dpi=getattr(config.plot_style, 'dpi', 150), # Use default dpi if not set
                           bbox_inches='tight')
                plt.close() # Close the R evolution plot figure
                self.log_progress(f"Saved R parameter evolution plot to {output_path_r}")

        except Exception as e:
             self.log_progress(f"Failed to plot R evolution: {e}", level='error')
             plt.close() # Ensure figure is closed even on error

        # --- Plot Photon Number Distributions ---
        # Check if this plotting is enabled in config
        plot_photon_dist = getattr(config, 'plot_photon_distribution', True) # Default to True if not set
        if not plot_photon_dist:
            self.log_progress("Photon distribution plotting is disabled.")
            return

        # Check if states are cached
        if 'states' in self._cache and 'times' in self._cache:
            # Use directly cached states from the simulation
            cached_states = self._cache['states']
            cached_times = self._cache['times']
            self.log_progress("Using cached states from simulation for photon distribution plots.")
        elif 'states_by_g' in self._cache:
            # Use states loaded from files
            states_by_g = self._cache['states_by_g']

            # Find the g_value with the most extreme R values
            max_r_diff = -1
            best_g = None

            for g_value, (scaled_t, r_values) in values.items():
                # Convert tuple keys to float if needed
                if isinstance(g_value, tuple) and len(g_value) == 2:
                    g_val = g_value[0]  # Extract the numeric g-value
                else:
                    g_val = g_value

                r_min = min(r_values)
                r_max = max(r_values)
                r_diff = r_max - r_min

                if r_diff > max_r_diff and g_val in states_by_g:
                    max_r_diff = r_diff
                    best_g = g_val

            if best_g is None:
                self.log_progress("No matching g-values found in states cache. Skipping photon distribution plots.", level='warning')
                return

            # Use the states for the selected g-value
            cached_states, cached_times = states_by_g[best_g]
            # Store in the cache in the format expected by the rest of the method
            self._cache['states'] = cached_states
            self._cache['times'] = cached_times
            self.log_progress(f"Using loaded states for g={best_g} for photon distribution plots.")
        else:
            self.log_progress("States or times not found in cache. Skipping photon distribution plots.", level='warning')
            return

        # Helper function for plotting distribution
        def _plot_distribution(n_vals, probs, plot_title, file_path, plot_config):
            try:
                fig, ax = plt.subplots(figsize=getattr(plot_config, 'figsize', (8, 5)))
                ax.bar(n_vals, probs, width=0.8, align='center', color='#1f77b4')
                ax.set_xlabel("Cavity Photon Number (n)")
                ax.set_ylabel("Probability P(n)")
                ax.set_title(plot_title, fontsize=10) # Slightly smaller title font
                # Ensure ticks cover the range of plotted n_vals
                if len(n_vals) > 0:
                    ax.set_xticks(np.arange(min(n_vals), max(n_vals)+1))
                    ax.set_xlim(min(n_vals) - 0.5, max(n_vals) + 0.5) # Set limits around bars
                ax.grid(getattr(plot_config, 'grid', True), axis='y', alpha=0.5)
                ax.set_ylim(bottom=0) # Probability cannot be negative
                # Add text annotation for sum of probabilities (as a sanity check)
                sum_probs = np.sum(probs)
                ax.text(0.95, 0.95, f'∑P(n) = {sum_probs:.4f}',
                        horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                plt.tight_layout()
                plt.savefig(file_path, dpi=getattr(plot_config, 'dpi', 150), bbox_inches='tight')
                plt.close(fig) # Close the figure associated with these axes
                self.log_progress(f"Saved photon distribution plot to {file_path}")
            except Exception as plot_err:
                self.log_progress(f"Failed to generate distribution plot '{plot_title}': {plot_err}", level='error')
                plt.close(fig) # Ensure figure is closed

        # Plot distributions for each g_value that was successfully plotted before
        for g_value in plotted_g_values:
            # Retrieve the correct data for this g_value
            scaled_t, r_values = values[g_value]

            self.log_progress(f"Generating photon distribution plots for g/ν={g_value}")

            # Double-check lengths against cached states
            if len(r_values) != len(cached_states) or len(scaled_t) != len(r_values):
                self.log_progress(f"Data length mismatch for g/ν={g_value} (R:{len(r_values)}, States:{len(cached_states)}). Skipping distribution plots.", level='warning')
                continue

            if len(r_values) < 1: # Need at least one point
                self.log_progress(f"No R values for g/ν={g_value}. Skipping distribution plots.", level='warning')
                continue

            # --- Find Indices ---
            try:
                idx_min_r = np.argmin(r_values)
                time_min_r = scaled_t[idx_min_r]
                val_min_r = r_values[idx_min_r]

                idx_max_r_non_initial = -1
                time_max_r = -1.0
                val_max_r = -np.inf
                if len(r_values) > 1:
                    # Find max R value excluding the first element (index 0)
                    max_r_val_non_initial = np.max(r_values[1:])
                    # Find the first index matching this max value (add 1 to shift index)
                    idx_max_r_non_initial = np.where(r_values[1:] == max_r_val_non_initial)[0][0] + 1
                    time_max_r = scaled_t[idx_max_r_non_initial]
                    val_max_r = r_values[idx_max_r_non_initial] # Use the actual value at the found index
                else:
                    self.log_progress(f"Only one time point for g/ν={g_value}. Cannot find max R excluding initial.", level='info')

                indices_to_plot = {
                    f'minR_t{time_min_r:.2f}': idx_min_r
                }
                if idx_max_r_non_initial != -1:
                    indices_to_plot[f'maxR_t{time_max_r:.2f}'] = idx_max_r_non_initial

            except IndexError as ie:
                 self.log_progress(f"IndexError finding min/max R index for g/ν={g_value}: {ie}. Skipping distribution plots.", level='error')
                 continue


            # --- Plot for selected indices ---
            plotted_indices = set() # Avoid plotting same index twice if min=max
            for label_key, idx in indices_to_plot.items():
                if idx in plotted_indices:
                    continue
                plotted_indices.add(idx)

                try:
                    state_full = cached_states[idx]
                    # Get cavity dimension Nb from the state (index 1)
                    dims = state_full.dims[0]
                    if len(dims) != 3:
                         self.log_progress(f"State {idx} for g/ν={g_value} does not have 3 subsystems. Skipping.", level='warning')
                         continue
                    N_b = dims[1]

                    # Trace out qubit (0) and vibrational (2) modes to get cavity state
                    rho_cavity = state_full.ptrace(1) # Keep only cavity (index 1)

                    # Get probabilities P(n) = diagonal elements
                    # Ensure probabilities are non-negative real numbers
                    probs = np.maximum(0.0, rho_cavity.diag().real)
                    # Renormalize slightly if needed due to numerical errors
                    sum_p = np.sum(probs)
                    if abs(sum_p - 1.0) > 1e-6:
                        self.log_progress(f"Renormalizing P(n) for state {idx}, g/ν={g_value}. Original sum={sum_p:.6f}", level='debug')
                        probs /= sum_p

                    n_vals = np.arange(N_b)

                    # Define title and filename
                    r_val_label = r_values[idx]
                    time_label_val = scaled_t[idx]
                    plot_label = "Lowest R" if "minR" in label_key else "Highest R (t>0)"
                    plot_title = f"Cavity Photon Distribution at {plot_label}\n(t={time_label_val:.2f}, R={r_val_label:.3f}) for g/ν={g_value}{title_suffix}"
                    output_filename = f'{base_filename}_photon_dist_{label_key}_g{g_value}{title_suffix}.png'
                    output_path = config.output_dir / output_filename

                    # Plot using helper function
                    _plot_distribution(n_vals, probs, plot_title, output_path, config.plot_style)

                except Exception as e:
                    self.log_progress(f"Error processing state for {label_key} (g/ν={g_value}, idx={idx}): {e}", level='error')