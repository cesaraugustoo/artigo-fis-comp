"""
Coherence Metric Module
=====================

This module provides the CoherenceMetric class for calculating quantum coherence
using the relative coherence measure.
"""

import logging
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, entropy_vn

from metrics.base import MetricCalculator
from config import MetricsConfig
from exceptions import MetricCalculationError, error_handler
from validators import StateValidator, StatePreparationError
from numerical_analysis import DiagnosticResult

# Setup logging
logger = logging.getLogger(__name__)

# class CoherenceMetric(MetricCalculator):
#     """Calculator for quantum coherence using the relative coherence measure.

#     The measure is defined as:
#         C(ρ) = S(Π[ρ]) - S(ρ)
#     where S(·) is the von Neumann entropy (base 2) and
#         Π[ρ] = ∑_i |i⟩⟨i| ρ |i⟩⟨i|
#     is the projection onto the energy (σ_z) eigenbasis.
#     When spontaneous emission is the dispersion mechanism, the process
#     both removes the off-diagonals and changes the populations.
#     """

#     @error_handler(MetricCalculationError)
#     def calculate(self, states: List[Qobj], times: np.ndarray,
#                   g_value: float) -> np.ndarray:
#         """Calculate quantum coherence evolution under spontaneous emission.

#         Coherence is measured as the difference between the von Neumann entropy
#         of the state after applying the spontaneous emission projection (which
#         removes the off-diagonals in the σ_z basis) and the original state's
#         von Neumann entropy:

#             C(ρ) = S(Π[ρ]) - S(ρ)

#         Args:
#             states: List of quantum states at different time points.
#             times: Array of time points.
#             g_value: Coupling strength value (for numerical analysis).

#         Returns:
#             Array of coherence values.

#         Raises:
#             MetricCalculationError: If calculation fails.
#         """
#         self.log_progress("Starting coherence calculation")

#         try:
#             # Validate input states.
#             self.log_progress("Validating input states")

#             validation = StateValidator.validate_state_lists(states, check_dims=True)
#             if not validation.is_valid:
#                 raise StatePreparationError(validation.message)

#             # Begin analysis for this g-value if enabled.
#             if self._num_analysis_enabled:
#                 self._analyzer.start_g_value_analysis(g_value)

#             # Initialize operators if not already done
#             if not self.operators:
#                 dims = states[0].dims[0]
#                 self.initialize_operators(dims[2], dims[1])  # N_a, N_b

#             # Define the basis states: eigenstates of σ_z.
#             basis_states = [self.operators.get_operator('e'), self.operators.get_operator('g')]

#             coherence_values = []
#             total_states = len(states)

#             for idx, state in enumerate(states, 1):
#                 if self._num_analysis_enabled:
#                     self._analyzer.analyze_state(state, f'step_{idx}_state')

#                 # Get the qubit reduced density matrix.
#                 rho_qubit = state.ptrace(0)

#                 if self._num_analysis_enabled:
#                     self._analyzer.analyze_state(rho_qubit, f'step_{idx}_reduced')

#                 # Apply the projection corresponding to spontaneous emission:
#                 # This removes the off-diagonals in the σ_z basis and reflects the effect
#                 # of the decay process on coherence.
#                 rho_dephased = sum(
#                     psi * psi.dag() * rho_qubit * psi * psi.dag()
#                     for psi in basis_states
#                 )

#                 # Compute the von Neumann entropies (base 2)
#                 entropy_dephased = entropy_vn(rho_dephased, base=2)
#                 entropy_rho = entropy_vn(rho_qubit, base=2)

#                 # Calculate the relative coherence measure.
#                 coherence = entropy_dephased - entropy_rho

#                 if self._num_analysis_enabled:
#                     self._analyzer.results[g_value].append(DiagnosticResult(
#                         operation=f'step_{idx}_coherence',
#                         timestamp=time.time(),
#                         values={
#                             'entropy_dephased': entropy_dephased,
#                             'entropy': entropy_rho,
#                             'coherence': coherence
#                         }
#                     ))

#                 coherence_values.append(coherence)

#                 if idx % (total_states // 10) == 0:
#                     self.log_progress(f"Processed {idx}/{total_states} states", 'debug')

#             # Save diagnostics if enabled.
#             if self._num_analysis_enabled:
#                 self.save_diagnostics()

#             return np.array(coherence_values)

#         except Exception as e:
#             self.log_progress(f"Coherence calculation failed: {str(e)}", 'error')
#             raise MetricCalculationError("Failed to calculate coherence") from e

# class CoherenceMetric(MetricCalculator):
#     """
#     Quantum coherence via relative entropy of coherence:
#         C(\rho) = S(\Pi[\rho]) - S(\rho),
#     where \Pi projects onto the chosen basis (removes off-diagonals).
#     """

#     @error_handler(MetricCalculationError)
#     def apply_config(self, config: MetricsConfig) -> None:
#         """Optional config for logging/analysis."""
#         if getattr(config, 'coherence_floor', None) is not None:
#             self.coherence_floor = config.coherence_floor
#         # could add more config options here

#     @error_handler(MetricCalculationError)
#     def calculate(
#         self,
#         states: List[Qobj],
#         times: np.ndarray,
#         g_value: float
#     ) -> np.ndarray:
#         """
#         Compute coherence C(\rho) = S(diag(\rho)) - S(\rho) for each state.
#         Assumes states already include physical dissipation.
#         Returns values in [0, 1] (for qubit).
#         """
#         self.log_progress("Starting coherence calculation")

#         # Validate inputs
#         val = StateValidator.validate_state_lists(states, check_dims=True)
#         if not val.is_valid:
#             raise StatePreparationError(val.message)

#         if self._num_analysis_enabled:
#             self._analyzer.start_g_value_analysis(g_value)

#         coherence_vals = []
#         total = len(states)
#         step = max(1, total // 10)

#         for idx, state in enumerate(states, start=1):
#             # Reduce to qubit
#             rho_qubit = state.ptrace(0)

#             if self._num_analysis_enabled:
#                 self._analyzer.analyze_state(rho_qubit, f'step_{idx}_reduced')

#             # Obtain diagonal probabilities in chosen basis
#             rho_mat = rho_qubit.full()
#             pops = np.real(np.diag(rho_mat))

#             # Post-dephasing entropy: Shannon on pops
#             S_dephased = -np.sum(pops * np.log2(pops + 1e-16))

#             # Original von Neumann entropy
#             S_original = entropy_vn(rho_qubit, base=2)

#             # Coherence metric
#             C = S_dephased - S_original
#             # Clamp to [0,1]
#             # C = max(0.0, min(C, 1.0))

#             coherence_vals.append(C)

#             if self._num_analysis_enabled:
#                 self._analyzer.results[g_value].append(DiagnosticResult(
#                     operation=f'step_{idx}_coherence',
#                     timestamp=time.time(),
#                     values={'S_dephased': S_dephased, 'S_original': S_original, 'coherence': C}
#                 ))

#             if idx % step == 0:
#                 self.log_progress(f"Processed {idx}/{total} states", level='debug')

#         if self._num_analysis_enabled:
#             self.save_diagnostics()

#         return np.array(coherence_vals)

class CoherenceMetric(MetricCalculator):
    """Calculator for quantum coherence using the relative coherence measure.

    The measure is defined as:
        C(ρ) = S(Π[ρ]) - S(ρ)
    where S(·) is the von Neumann entropy (base 2) and
        Π[ρ] = ∑_i |i⟩⟨i| ρ |i⟩⟨i|
    is the projection onto the specified basis (here, qubit |g>, |e>).
    """

    @error_handler(MetricCalculationError)
    def calculate(self, states: List[Qobj], times: np.ndarray,
                  g_value: float) -> np.ndarray:
        """Calculate quantum coherence evolution.

        Coherence is measured as C(ρ) = S(Π[ρ]) - S(ρ), where ρ is the
        reduced density matrix of the qubit subsystem and Π is the dephasing
        operation in the σ_z (|g>, |e>) basis.

        Args:
            states: List of full quantum states (e.g., [qubit, cavity, vib]).
            times: Array of time points.
            g_value: Coupling strength value (for numerical analysis).

        Returns:
            Array of coherence values C(ρ_qubit).

        Raises:
            MetricCalculationError: If calculation fails.
            StatePreparationError: If input states are invalid.
        """
        self.log_progress(f"Starting coherence calculation for g/ν = {g_value}")
        start_time = time.time()

        try:
            if not states:
                self.log_progress("Input state list is empty. Returning empty array.", 'warning')
                return np.array([])

            # Validate input states.
            self.log_progress("Validating input states")
            validation = StateValidator.validate_state_lists(states, check_dims=True)
            if not validation.is_valid:
                raise StatePreparationError(validation.message)

            # Begin analysis for this g-value if enabled.
            if self._num_analysis_enabled:
                self._analyzer.start_g_value_analysis(g_value)

            # Initialize operators if not already done
            # Note: Only qubit operators |g>, |e> are strictly needed here.
            if not self.operators:
                dims = states[0].dims[0]
                # Assuming [qubit, cavity, vibrational]
                if len(dims) < 3:
                     raise StatePreparationError(f"Expected at least 3 subsystems, got dims {dims}")
                # Initialize based on other subsystems for consistency? Or just qubit?
                # Let's assume the base class handles this appropriately.
                self.initialize_operators(dims[2], dims[1]) # N_a, N_b

            # Define the basis states: eigenstates of σ_z.      
            basis_states = [self.operators.get_operator('e'), self.operators.get_operator('g')]

            coherence_values = []
            total_states = len(states)
            step_interval = max(1, total_states // 10)

            for idx, state in enumerate(states):
                current_step = idx + 1
                log_prefix = f"State {current_step}/{total_states}: "
                self.log_progress(f"{log_prefix}Processing", level='debug')

                if self._num_analysis_enabled:
                    self._analyzer.analyze_state(state, f'step_{current_step}_full_state')

                # Get the qubit reduced density matrix (subsystem index 0)
                try:
                    rho_qubit = state.ptrace(0)
                except Exception as ptrace_err:
                    self.log_progress(f"{log_prefix}Failed to perform partial trace: {ptrace_err}", 'error')
                    raise MetricCalculationError(f"Partial trace failed for state {idx}") from ptrace_err

                # --- Sanity check rho_qubit ---
                if rho_qubit.dims != [[2], [2]]:
                     self.log_progress(f"{log_prefix}Error: rho_qubit has unexpected dims {rho_qubit.dims}", 'error')
                     raise MetricCalculationError(f"Incorrect dimensions for rho_qubit at step {idx}")
                trace_val = rho_qubit.tr()
                if not np.isclose(trace_val, 1.0, atol=1e-7):
                     self.log_progress(f"{log_prefix}Warning: rho_qubit trace is {trace_val:.6f} (should be 1.0)", 'warning')
                # --- End Sanity Check ---

                if self._num_analysis_enabled:
                    self._analyzer.analyze_state(rho_qubit, f'step_{current_step}_reduced_qubit')

                # --- Calculate S(ρ_qubit) ---
                try:
                    # Use qutip's entropy_vn, specify base 2
                    entropy_rho = entropy_vn(rho_qubit, base=2)
                    self.log_progress(f"{log_prefix}S(ρ_qubit) = {entropy_rho:.6f}", level='debug')
                except Exception as e:
                    self.log_progress(f"{log_prefix}Failed to calculate S(ρ_qubit): {e}", 'error')
                    raise MetricCalculationError(f"Entropy calculation failed for rho_qubit at step {idx}") from e


                # --- Calculate S(Π[ρ_qubit]) ---
                # Π[ρ] = ∑_i |i⟩⟨i| ρ |i⟩⟨i| = diag(ρ_ii)
                # S(Π[ρ]) = -∑_i ρ_ii log2(ρ_ii)

                # OPTIMIZED APPROACH: Directly use diagonal elements
                try:
                    diag_elements = rho_qubit.diag()

                    # Filter out zero probabilities to avoid log2(0) -> NaN issues
                    # Use a small tolerance for floating point comparisons
                    prob_tolerance = 1e-15
                    non_zero_diag = diag_elements[diag_elements > prob_tolerance]

                    if len(non_zero_diag) > 0:
                        # Calculate entropy sum: - Σ p*log2(p)
                        entropy_dephased = -np.sum(non_zero_diag * np.log2(non_zero_diag))
                    else:
                        # Handle cases where state might be invalid or numerically zero
                        entropy_dephased = 0.0
                        if np.any(diag_elements < -prob_tolerance) or not np.isclose(sum(diag_elements), 1.0):
                             self.log_progress(f"{log_prefix}Warning: Invalid diagonal elements {diag_elements} for entropy calculation.", 'warning')

                    self.log_progress(f"{log_prefix}S(Π[ρ_qubit]) = {entropy_dephased:.6f} (optimized)", level='debug')

                except Exception as e:
                    self.log_progress(f"{log_prefix}Failed to calculate S(Π[ρ_qubit]) using optimized method: {e}", 'error')
                    raise MetricCalculationError(f"Entropy calculation failed for dephased state at step {idx}") from e

                # --- Original (Less Efficient) Approach ---
                # rho_dephased = sum(
                #     psi * psi.dag() * rho_qubit * psi * psi.dag()
                #     for psi in basis_states
                # )
                # entropy_dephased_orig = entropy_vn(rho_dephased, base=2)
                # self.log_progress(f"{log_prefix}S(Π[ρ_qubit]) = {entropy_dephased_orig:.6f} (original method)", level='debug')
                # # Add a check: np.isclose(entropy_dephased, entropy_dephased_orig)
                # --- End Original Approach ---


                # --- Calculate the relative coherence measure ---
                coherence = entropy_dephased - entropy_rho
                # Coherence should be non-negative
                if coherence < -1e-9: # Allow for small numerical errors
                    self.log_progress(f"{log_prefix}Warning: Calculated coherence is negative ({coherence:.4f}). Clamping to 0.", 'warning')
                    coherence = 0.0
                else:
                    # Ensure it's exactly zero if very close
                    coherence = max(0.0, coherence)


                if self._num_analysis_enabled:
                    # Assuming DiagnosticResult exists and analyzer handles it
                    self._analyzer.results[g_value].append(DiagnosticResult(
                        operation=f'step_{current_step}_coherence',
                        timestamp=time.time(),
                        values={
                            'entropy_dephased': entropy_dephased,
                            'entropy_rho': entropy_rho,
                            'coherence': coherence
                        }
                    ))

                coherence_values.append(coherence)

                if current_step % step_interval == 0 or current_step == total_states:
                    self.log_progress(f"Processed {current_step}/{total_states} states")

            # Save diagnostics if enabled.
            if self._num_analysis_enabled:
                self.save_diagnostics() # Assuming this method exists

            end_time = time.time()
            self.log_progress(f"Coherence calculation completed in {end_time - start_time:.2f} seconds")
            return np.array(coherence_values)

        except StatePreparationError as e:
             self.log_progress(f"Input state validation failed: {str(e)}", 'error')
             raise # Re-raise specific error
        except MetricCalculationError as e:
             self.log_progress(f"Metric calculation failed: {str(e)}", 'error')
             raise # Re-raise specific error
        except Exception as e:
            self.log_progress(f"An unexpected error occurred during coherence calculation: {str(e)}", 'critical')
            # import traceback; traceback.print_exc() # For debugging
            raise MetricCalculationError("Unexpected failure in coherence calculation") from e

    @error_handler(MetricCalculationError)
    def plot(self, base_filename: str,
             values: Dict[float, Tuple[np.ndarray, np.ndarray]],
             config: MetricsConfig, title_suffix: str = '') -> None:
        """Plot coherence evolution.

        Creates a plot showing how the quantum coherence varies over time
        for different coupling strengths.

        Args:
            base_filename: Base name for output files.
            values: Dictionary mapping coupling strengths to results.
            config: Metrics configuration.
            title_suffix: Optional suffix for plot title.
        """
        if not config.plot:
            return

        plt.figure(figsize=config.plot_style.figsize)

        for g_value, (scaled_t, coherence) in values.items():
            plt.plot(scaled_t, coherence, label=f'g/ν={g_value}')

        time_label = 'gt' if config.use_gt_scale else 'νt'
        plt.xlabel(time_label)
        plt.ylabel('Coherence C(ρ)')
        plt.title(f'Quantum Coherence Evolution{title_suffix}')
        plt.grid(config.plot_style.grid)

        # Add reference lines.
        plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # Reference for pure states.
        plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)  # Reference for maximally mixed states.

        plt.legend()

        output_path = config.output_dir / f'{base_filename}_coherence{title_suffix}.png'
        plt.savefig(output_path, dpi=config.plot_style.dpi, bbox_inches='tight')
        plt.close()
