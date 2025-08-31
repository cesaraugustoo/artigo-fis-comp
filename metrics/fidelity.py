"""
Fidelity Metric Module
=====================

This module provides the FidelityMetric class for calculating fidelity between
full and RWA evolution states.
"""

import logging
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, fidelity

from metrics.base import MetricCalculator
from config import MetricsConfig
from exceptions import MetricCalculationError, error_handler
from validators import StateValidator, StatePreparationError
from numerical_analysis import DiagnosticResult

# Setup logging
logger = logging.getLogger(__name__)

class FidelityMetric(MetricCalculator):
    """Calculator for fidelity between full and RWA evolution states."""

    @error_handler(MetricCalculationError)
    def calculate(self, states_full: List[Qobj], states_rwa: List[Qobj],
                 times: np.ndarray, g_value: float) -> np.ndarray:
        """Calculate fidelity between full and RWA evolution states.

        The fidelity measure quantifies how well the RWA approximates the
        full evolution. A fidelity of 1 indicates perfect agreement.

        Args:
            states_full: States from full Hamiltonian evolution
            states_rwa: States from RWA evolution
            times: Array of time points

        Returns:
            Array of fidelity values

        Raises:
            MetricCalculationError: If calculation fails
            ValueError: If input states have different lengths
        """
        self.log_progress("Starting fidelity calculation between full and RWA states")

        try:
            if len(states_full) != len(states_rwa):
                raise ValueError("Full and RWA state lists must have same length")

            # Validate states
            self.log_progress("Validating input states")
            states_full, states_rwa = StateValidator.validate_and_prepare_states(
                states_full, states_rwa, times)

            # Start analysis for this g-value if enabled
            if self._num_analysis_enabled:
                self._analyzer.start_g_value_analysis(g_value)

            fidelities = []
            total_states = len(states_full)

            for idx, (state_full, state_rwa) in enumerate(zip(states_full, states_rwa), 1):
                if self._num_analysis_enabled:
                    # Analyze input states
                    self._analyzer.analyze_state(state_full, f'step_{idx}_full')
                    self._analyzer.analyze_state(state_rwa, f'step_{idx}_rwa')

                # Calculate fidelity
                fid = float(fidelity(state_full, state_rwa).real)

                if self._num_analysis_enabled:
                    self._analyzer.results[g_value].append(DiagnosticResult(
                        operation=f'step_{idx}_fidelity',
                        timestamp=time.time(),
                        values={'fidelity': fid}
                    ))

                fidelities.append(fid)

                if idx % (total_states // 10) == 0:
                    self.log_progress(f"Processed {idx}/{total_states} states", 'debug')

            # Save diagnostics if enabled
            if self._num_analysis_enabled:
                self.save_diagnostics()

            return np.array(fidelities)

        except Exception as e:
            self.log_progress(f"Fidelity calculation failed: {str(e)}", 'error')
            raise MetricCalculationError("Failed to calculate fidelity") from e

    @error_handler(MetricCalculationError)
    def plot(self, base_filename: str,
            values: Dict[float, Tuple[np.ndarray, np.ndarray]],
            config: MetricsConfig, title_suffix: str = '') -> None:
        """Plot fidelity evolution.

        Creates a plot showing how the fidelity between full and RWA
        evolution varies over time for different coupling strengths.

        Args:
            base_filename: Base name for output files
            values: Dictionary mapping coupling strengths to results
            config: Metrics configuration
            title_suffix: Optional suffix for plot title
        """
        if not config.plot:
            return

        plt.figure(figsize=config.plot_style.figsize)

        # Extract unique g-values from keys (which might be tuples or floats)
        unique_g_values = set()
        for key in values.keys():
            if isinstance(key, tuple) and len(key) == 2:
                # This is a (g_value, ham_type) tuple
                unique_g_values.add(key[0])
            else:
                # This is just a g_value
                unique_g_values.add(key)

        g_values = sorted(unique_g_values)

        # For fidelity plots, we're always plotting the full vs RWA comparison
        for g_value in g_values:
            # Find all keys that match this g_value
            matching_keys = [k for k in values.keys() if (k == g_value) or
                            (isinstance(k, tuple) and len(k) == 2 and k[0] == g_value)]

            # Group keys by hamiltonian type
            full_keys = [k for k in matching_keys if isinstance(k, tuple) and len(k) == 2 and
                        (k[1] == 'h_plus' or k[1] == 'h_minus')]
            standard_keys = [k for k in matching_keys if not isinstance(k, tuple)]

            # Determine which key to use
            if full_keys:
                # Prefer full Hamiltonian data if available
                key = full_keys[0]
                self.log_progress(f"Using full Hamiltonian data for g={g_value}", level='debug')
            elif standard_keys:
                # Fall back to standard keys if no full data
                key = standard_keys[0]
                self.log_progress(f"Using standard data for g={g_value}", level='debug')
            else:
                # Skip this g_value if no matching keys
                self.log_progress(f"No data found for g={g_value}", level='warning')
                continue

            scaled_t, fidelities = values[key]
            plt.plot(scaled_t, fidelities, label=f'g/nu={g_value}')

        time_label = 'gt' if config.use_gt_scale else 'νt'
        plt.xlabel(time_label)
        plt.ylabel('Fidelity')
        plt.title(f'State Fidelity Evolution{title_suffix}')
        plt.grid(config.plot_style.grid)
        # plt.legend() # Commented because the information goes at the subtitle

        # Add reference line for perfect fidelity
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

        # Add explanatory text (commented for better viewing)
        # plt.text(0.02, 0.02,
        #         'Fidelity = 1: Perfect RWA agreement\n' +
        #         'Fidelity < 1: RWA deviation',
        #         transform=plt.gca().transAxes,
        #         bbox=dict(facecolor='white', alpha=0.8))

        output_path = config.output_dir / f'{base_filename}_fidelity{title_suffix}.png'
        plt.savefig(output_path,
                   dpi=config.plot_style.dpi,
                   bbox_inches='tight')
        plt.close()
