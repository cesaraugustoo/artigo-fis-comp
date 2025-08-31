"""
Base Metric Calculator Module
============================

This module provides the abstract base class for all quantum metric calculators.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from config import MetricsConfig
from exceptions import MetricError, MetricCalculationError, error_handler
from operators import QuantumOperators
from numerical_analysis import DiagnosticResult, NumericalAnalyzer

# Setup logging
logger = logging.getLogger(__name__)

class MetricCalculator(ABC):
    """Abstract base class for quantum metric calculators."""

    def __init__(self):
        """Initialize metric calculator."""
        self._cache = {}
        self.metric_name = self.__class__.__name__.replace('Metric', '').lower()
        self._analyzer = None
        self._num_analysis_enabled = False
        self.operators = None

    def setup_numerical_analysis(self, enabled: bool, output_dir: Path) -> None:
        """Setup numerical analysis configuration.

        Args:
            enabled: Whether to enable numerical analysis
            output_dir: Directory for saving diagnostic results
        """
        self._num_analysis_enabled = enabled
        if enabled:
            self._analyzer = NumericalAnalyzer(debug=False)
            self._diagnostic_dir = output_dir / 'diagnostics'
            self._diagnostic_dir.mkdir(parents=True, exist_ok=True)

    def save_diagnostics(self) -> None:
        """Save diagnostic results and generate plots if numerical analysis is enabled."""
        if not self._num_analysis_enabled or not self._analyzer:
            return

        try:
            # Get diagnostic summary
            summary = self._analyzer.get_summary()
            self.log_progress("\nNumerical Analysis Summary:")
            for op_type, stats in summary.items():
                self.log_progress(f"{op_type}:")
                self.log_progress(f"  Max error: {stats['max_error']:.2e}")
                self.log_progress(f"  Mean error: {stats['mean_error']:.2e}")
                if stats.get('max_condition', 0) > 0:
                    self.log_progress(f"  Max condition number: {stats['max_condition']:.2e}")

            # Save numerical results
            diagnostic_file = self._diagnostic_dir / f'{self.metric_name}_analysis.npy'
            self._analyzer.save_results(diagnostic_file)
            self.log_progress(f"Diagnostics saved to {diagnostic_file}")

            # Generate diagnostic plots
            self.log_progress("Generating diagnostic plots...")
            self._analyzer.plot_diagnostics(self._diagnostic_dir)
            self.log_progress("Diagnostic plots generated")

        except Exception as e:
            self.log_progress(f"Warning: Could not save diagnostics: {str(e)}", 'warning')

    def initialize_operators(self, N_a: int, N_b: int) -> None:
        """Initialize quantum operators with given dimensions.

        Args:
            N_a: Dimension of vibrational mode Hilbert space
            N_b: Dimension of cavity field Hilbert space
        """
        self.operators = QuantumOperators(N_a, N_b)
        self.operators.create_basic_operators()
        self.operators.create_composite_operators()

    @abstractmethod
    def calculate(self, *args, **kwargs) -> np.ndarray:
        """Calculate metric values."""
        pass

    @abstractmethod
    def plot(self, base_filename: str,
            values: Dict[float, Tuple[np.ndarray, np.ndarray]],
            config: MetricsConfig, title_suffix: str = '') -> None:
        """Plot metric results.

        Args:
            base_filename: Base name for output files
            values: Dictionary mapping coupling strengths to (times, metric_values) tuples
                   Note: metric_values can be either 1D or 2D arrays
            config: Metrics configuration
            title_suffix: Optional suffix for plot title
        """
        pass

    def _ensure_2d_array(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D by reshaping if necessary.

        Args:
            arr: Input array that might be 1D or 2D

        Returns:
            2D array with shape (n, 1) if input was 1D, or original array if already 2D
        """
        if len(arr.shape) == 1:
            self.log_progress(f"Converting 1D array of shape {arr.shape} to 2D", level='debug')
            return arr.reshape(-1, 1)
        return arr

    def log_progress(self, message: str, level: str = 'info'):
        """Log progress message with metric context."""
        log_func = getattr(logger, level)
        log_func(f"[{self.metric_name}] {message}")

    def set_states(self, states_by_g: Dict[float, Tuple[List, np.ndarray]]) -> None:
        """Set states and times for additional plots.

        This method is used when loading metric data from file but still wanting to
        generate additional plots that require the original states.

        Args:
            states_by_g: Dictionary mapping g-values to (states, times) tuples
        """
        self._cache['states_by_g'] = states_by_g
        self.log_progress(f"States for {len(states_by_g)} g-values stored in cache for additional plots")

    def clear_cache(self) -> None:
        """Clear calculator cache and operator cache."""
        self._cache.clear()
        if self.operators:
            self.operators.clear_cache()
        if self._analyzer:
            self._analyzer.clear()
