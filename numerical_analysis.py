"""
Numerical Analysis Module
======================

This module provides tools for analyzing numerical errors and stability in quantum
calculations. It can be used to track various error metrics and diagnose numerical
issues in quantum simulations.

Classes:
    NumericalAnalyzer: Main class for numerical analysis
    DiagnosticResult: Container for diagnostic results
    
Functions:
    save_diagnostics: Utility function for saving diagnostic data
"""

import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from qutip import Qobj, expect

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticResult:
    """Container for diagnostic results."""
    operation: str
    timestamp: float
    values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of diagnostic result."""
        result = f"Diagnostic for {self.operation}:\n"
        for key, value in self.values.items():
            result += f"  {key}: {value:.2e}\n"
        return result

class NumericalAnalyzer:
    """Analyzer class for tracking numerical errors and stability."""
    
    def __init__(self, debug: bool = False):
        """Initialize numerical analyzer."""
        self.debug = debug
        self.results = {}  # Dictionary keyed by g_value
        self._operation_types = set()
        self._current_g = None  # Track current g_value
    
    def start_g_value_analysis(self, g_value: float):
        """Initialize analysis for a new g-value."""
        self.results[g_value] = []
        self._current_g = g_value
        
    def _store_result(self, result: DiagnosticResult):
        """Store result for current g_value."""
        if self._current_g is None:
            raise ValueError("No g-value set. Call start_g_value_analysis first.")
        self.results[self._current_g].append(result)
        
    def analyze_expectation(self, state: Qobj, operator: Qobj,
                          operation_name: str) -> DiagnosticResult:
        """Analyze numerical stability of expectation value calculation."""
        try:
            # Calculate expectation value
            expval = expect(operator, state)
            
            # Enhanced stability checks
            rel_error = abs(expval.imag) / (abs(expval.real) + 1e-15)
            operator_norm = operator.norm()
            state_norm = state.norm()
            condition_number = operator_norm * state_norm
            
            # Check for potential numerical instability
            stability_warning = False
            if condition_number > 1e4:
                stability_warning = True
                logging.warning(f"High condition number ({condition_number:.2e}) "
                              f"for operation {operation_name}")
            
            result = DiagnosticResult(
                operation=operation_name,
                timestamp=time.time(),
                values={
                    'value_real': float(expval.real),
                    'value_imag': float(expval.imag),
                    'relative_error': float(rel_error),
                    'condition_number': float(condition_number),
                    'stability_warning': stability_warning
                }
            )
            
            self._operation_types.add('expectation')
            self._store_result(result)
            return result
            
        except Exception as e:
            logging.error(f"Error in expectation analysis: {str(e)}")
            raise
    
    def analyze_state(self, state: Qobj, operation_name: str) -> DiagnosticResult:
        """Analyze numerical properties of a quantum state."""
        try:
            trace = state.tr()
            herm_error = (state - state.dag()).norm('max')
            
            result = DiagnosticResult(
                operation=operation_name,
                timestamp=time.time(),
                values={
                    'trace_error': abs(trace - 1.0),
                    'hermiticity_error': float(herm_error),
                    'purity': abs((state * state).tr())
                }
            )
            
            self._operation_types.add('state')
            self._store_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in state analysis: {str(e)}")
            raise
            
    def analyze_variance(self, mean: float, squared_mean: float,
                        operation_name: str) -> DiagnosticResult:
        """Analyze numerical stability of variance calculation."""
        try:
            variance = squared_mean - mean**2
            
            rel_error_var = abs(variance) / (abs(squared_mean) + 1e-15)
            rel_error_mean = abs(mean) / (abs(squared_mean) + 1e-15)
            condition_number = squared_mean / (abs(mean) + 1e-15)
            
            result = DiagnosticResult(
                operation=operation_name,
                timestamp=time.time(),
                values={
                    'mean': float(mean),
                    'squared_mean': float(squared_mean),
                    'variance': float(variance),
                    'relative_error_variance': float(rel_error_var),
                    'relative_error_mean': float(rel_error_mean),
                    'condition_number': float(condition_number)
                }
            )
            
            self._operation_types.add('variance')
            self._store_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in variance analysis: {str(e)}")
            raise
            
    def analyze_integration(self, values: np.ndarray, dx: float,
                          operation_name: str) -> DiagnosticResult:
        """Analyze numerical stability of integration."""
        try:
            integral = np.sum(values) * dx
            expected_norm = 1.0
            rel_error = abs(integral - expected_norm) / (abs(expected_norm) + 1e-15)
            
            edge_values = np.max(np.abs(values[[0, -1]]))
            truncation_error = edge_values * dx
            
            max_value = float(np.max(np.abs(values)))
            min_value = float(np.min(np.abs(values)))
            mean_value = float(np.mean(np.abs(values)))
            
            result = DiagnosticResult(
                operation=operation_name,
                timestamp=time.time(),
                values={
                    'integral': float(integral),
                    'relative_error': float(rel_error),
                    'truncation_error': float(truncation_error),
                    'max_value': max_value,
                    'min_value': min_value,
                    'mean_value': mean_value,
                    'value_range': max_value - min_value
                }
            )
            
            self._operation_types.add('integration')
            self._store_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in integration analysis: {str(e)}")
            raise

    def plot_diagnostics(self, output_path: Path) -> None:
        """Create comprehensive diagnostic plots for all g-values."""
        if not self.results:
            return
            
        for g_value, g_results in self.results.items():
            # Group results by operation type
            grouped_results = {}
            for result in g_results:
                op_type = result.operation.split('_')[0]
                if op_type not in grouped_results:
                    grouped_results[op_type] = []
                grouped_results[op_type].append(result)
            
            # Create plots for each operation type
            for op_type, results in grouped_results.items():
                self._plot_operation_diagnostics(op_type, results, output_path, g_value)
    
    def _plot_operation_diagnostics(self, op_type: str, results: List[DiagnosticResult],
                                  output_path: Path, g_value: float) -> None:
        """Plot diagnostics for a specific operation type and g-value."""
        # Get all metrics for this operation type
        metrics = set()
        for result in results:
            metrics.update(result.values.keys())
        
        # Create a grid of plots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each metric
        for idx, metric in enumerate(sorted(metrics)):
            ax = axes[idx]
            values = [r.values.get(metric, np.nan) for r in results]
            times = range(len(values))
            
            ax.plot(times, values, 'b-', linewidth=1)
            
            if 'error' in metric.lower():
                valid_values = np.array(values)[~np.isnan(values) & (np.array(values) > 0)]
                if len(valid_values) > 0:
                    ax.set_yscale('log')
                    ax.axhline(y=1e-8, color='g', linestyle=':', label='Good')
                    ax.axhline(y=1e-4, color='y', linestyle=':', label='Warning')
                    ax.axhline(y=1e-2, color='r', linestyle=':', label='Critical')
                    ax.legend()
            else:
                ax.set_yscale('linear')
                
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.grid(True)
        
        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'{op_type.title()} Diagnostics (g/ν = {g_value})')
        plt.tight_layout()
        
        plot_path = output_path / f'{op_type}_diagnostics_g{g_value}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    def clear(self) -> None:
        """Clear all diagnostic results."""
        self.results.clear()
        self._operation_types.clear()
        self._current_g = None

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of diagnostics."""
        summary = {}
        
        for g_results in self.results.values():
            for result in g_results:
                op_type = result.operation.split('_')[0]
                if op_type not in summary:
                    summary[op_type] = {
                        'max_error': 0.0,
                        'mean_error': 0.0,
                        'count': 0
                    }
                
                stats = summary[op_type]
                stats['count'] += 1
                
                for key, value in result.values.items():
                    if 'error' in key.lower():
                        stats['max_error'] = max(stats['max_error'], value)
                        stats['mean_error'] += value
        
        # Calculate means
        for stats in summary.values():
            if stats['count'] > 0:
                stats['mean_error'] /= stats['count']
        
        return summary

    def save_results(self, output_file: Path) -> None:
        """Save numerical analysis results."""
        # Convert results to saveable format
        data = []
        for g_value, g_results in self.results.items():
            for result in g_results:
                data.append({
                    'g_value': g_value,
                    'operation': result.operation,
                    'timestamp': result.timestamp,
                    'values': result.values
                })
        
        np.save(output_file, data)