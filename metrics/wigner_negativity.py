"""
Wigner Negativity Metric Module
=============================

This module provides the WignerNegativityMetric class for calculating Wigner
function negativity.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, wigner

from metrics.base import MetricCalculator
from config import MetricsConfig
from exceptions import MetricCalculationError, error_handler

# Setup logging
logger = logging.getLogger(__name__)

class WignerNegativityMetric(MetricCalculator):
    """
    Calculates the Wigner function negativity for the vibrational mode.

    The Wigner negativity is a measure of non-classicality, defined as:
    N[ρ] = ∫ |W(α)| d²α - 1

    This implementation computes the integral numerically over a discrete grid.
    """

    def __init__(self):
        super().__init__()
        self.use_parallel = True
        self.max_workers = os.cpu_count()
        self._grid_created = False

    @error_handler(MetricCalculationError)
    def calculate(self, states: List[Qobj], times: np.ndarray, g_value: float) -> np.ndarray:
        """
        Compute Wigner negativity for a series of time-evolved states.
        """
        self.log_progress(f"Starting Wigner negativity calculation for g/ν = {g_value}")
        if not states:
            return np.array([])

        # Cache states for plotting later
        self._cache.setdefault('states_by_g', {})[g_value] = (states, times)

        # Create phase-space grid once
        if not self._grid_created:
            # The order is [qubit, cavity, vibration], so index is 2
            vib_dim = states[0].dims[0][2]
            self._create_wigner_grid(vib_dim)
            self._grid_created = True

        # Choose execution path
        if self.use_parallel and len(states) > 1:
            negativities = self._calculate_parallel(states)
        else:
            negativities = self._calculate_sequential(states)

        # Check for truncation issues on the final state
        final_rho_vib = states[-1].ptrace(2)
        self._check_truncation(final_rho_vib)

        return np.array(negativities)

    def _create_wigner_grid(self, vib_dim: int, grid_span: float = 6.0, points: int = 11):
        """
        Creates a uniform phase-space grid.
        A larger span can capture highly displaced states, while more points
        increase integration accuracy at the cost of computation time.
        """
        self.log_progress(f"Creating Wigner grid: span={grid_span}, points={points}")
        x = np.linspace(-grid_span, grid_span, points)
        y = x.copy()
        self._cache['xvec'] = x
        self._cache['yvec'] = y
        self._cache['dx'] = x[1] - x[0]
        self._cache['dy'] = y[1] - y[0]
        self._cache['grid_span'] = grid_span

    def _calculate_sequential(self, states: List[Qobj]) -> List[float]:
        """Process states sequentially."""
        return [self._process_single_state(state) for state in states]

    def _calculate_parallel(self, states: List[Qobj]) -> List[float]:
        """Process states in parallel."""
        results = [None] * len(states)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._process_single_state, state): i for i, state in enumerate(states)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.log_progress(f"Error processing state {index}: {e}", 'error')
                    results[index] = np.nan # Mark as failed
        return results

    def _process_single_state(self, state: Qobj) -> float:
        """Compute negativity for a single quantum state."""
        rho_vib = state.ptrace(2) # Trace out qubit and cavity
        W = wigner(rho_vib, self._cache['xvec'], self._cache['yvec'])
        
        # Numerical integration: sum |W(α)| d²α
        integral_abs = np.sum(np.abs(W)) * self._cache['dx'] * self._cache['dy']
        
        # Negativity N = integral - 1
        negativity = integral_abs - 1.0
        
        # Return non-negative result
        return max(0.0, negativity)

    def _check_truncation(self, rho_vib: Qobj, threshold: float = 1e-6):
        """
        Logs a warning if the population at the edge of the Hilbert space
        is above a given threshold, indicating insufficient truncation.
        """
        edge_population = rho_vib.diag()[-1]
        if edge_population > threshold:
            self.log_progress(
                f"Warning: Population at the edge of the vibrational Hilbert space "
                f"is {edge_population:.2e}, which is above the threshold of {threshold}. "
                f"Consider increasing the vibrational dimension 'N_a'.",
                level='warning'
            )

    def plot(self, base_filename: str, values: Dict[float, Tuple[np.ndarray, np.ndarray]],
             config: MetricsConfig, title_suffix: str = '') -> None:
        """
        Plots Wigner negativity evolution and snapshots.
        """
        if not config.plot or not values:
            return

        self._plot_negativity_evolution(values, base_filename, config, title_suffix)
        self._plot_wigner_snapshots(base_filename, values, config, title_suffix)

    def _plot_negativity_evolution(self, values: Dict[float, Tuple[np.ndarray, np.ndarray]],
                                   base_filename: str, config: MetricsConfig, title_suffix: str):
        """Plots the time evolution of Wigner negativity."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for g_val, (times, neg) in values.items():
            ax.plot(times, neg, label=f"g/ν = {g_val}")

        ax.set_xlabel('Time')
        ax.set_ylabel('Wigner Negativity')
        ax.set_title(f'Wigner Negativity Evolution{title_suffix}')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)

        filename = config.output_dir / f"{base_filename}_wigner_negativity{title_suffix}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        self.log_progress(f"Saved negativity evolution plot to {filename}")

    def _plot_wigner_snapshots(self, base_filename: str, values: Dict[float, Tuple[np.ndarray, np.ndarray]],
                               config: MetricsConfig, title_suffix: str):
        """Plots Wigner function snapshots at key time points."""
        for g_val, (times, neg) in values.items():
            if 'states_by_g' not in self._cache or g_val not in self._cache['states_by_g']:
                continue

            states, _ = self._cache['states_by_g'][g_val]
            
            # Find index of maximum negativity
            idx_max_neg = np.argmax(neg)
            
            indices = {'initial': 0, 'max_negativity': idx_max_neg, 'final': len(states) - 1}

            for name, idx in indices.items():
                state = states[idx]
                rho_vib = state.ptrace(2)
                W = wigner(rho_vib, self._cache['xvec'], self._cache['yvec'])
                
                vmax = np.max(np.abs(W))
                
                fig, ax = plt.subplots(figsize=(8, 6))
                contour = ax.contourf(self._cache['xvec'], self._cache['yvec'], W, 100, cmap='coolwarm',
                                      vmin=-vmax, vmax=vmax)
                ax.contour(self._cache['xvec'], self._cache['yvec'], W, levels=[0], colors='black', linewidths=1)
                fig.colorbar(contour, label='W(α)')
                ax.set_title(f'Wigner Function at {name} (t={times[idx]:.2f}) for g/ν={g_val}{title_suffix}')
                ax.set_xlabel('Re(α)')
                ax.set_ylabel('Im(α)')
                ax.set_aspect('equal')

                filename = config.output_dir / f"{base_filename}_wigner_g{g_val}_{name}{title_suffix}.png"
                fig.savefig(filename, dpi=300)
                plt.close(fig)
                self.log_progress(f"Saved Wigner snapshot for g={g_val} at {name} to {filename}")