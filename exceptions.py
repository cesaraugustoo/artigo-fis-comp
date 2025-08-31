"""
Custom Exceptions Module
======================

This module defines the custom exceptions hierarchy for the quantum simulation project.
These exceptions provide specific error types for different aspects of the simulation,
making error handling and debugging more precise.

Exception Hierarchy:
    QuantumSimError
    ├── ConfigurationError
    │   - Invalid parameter values
    │   - Missing required settings
    ├── OperatorError
    │   - Operator creation failures
    │   - Invalid operator operations
    ├── StatePreparationError
    │   - Invalid state parameters
    │   - State preparation failures
    ├── HamiltonianError
    │   - Hamiltonian construction issues
    │   - Invalid coupling parameters
    ├── SimulationError
    │   ├── TimeEvolutionError
    │   │   - Integration failures
    │   │   - Time step issues
    │   └── ConvergenceError
    │       - Numerical method failures
    │       - Convergence failures
    └── MetricError
        ├── MetricCalculationError
        │   - Calculation failures
        │   - Invalid metric parameters
        └── MetricVisualizationError
            - Plotting failures
            - Visualization issues
"""

class QuantumSimError(Exception):
    """Base exception for all quantum simulation errors."""
    def __init__(self, message: str = "An error occurred in the quantum simulation"):
        self.message = message
        super().__init__(self.message)

class ConfigurationError(QuantumSimError):
    """Exception raised for errors in configuration parameters."""
    def __init__(self, message: str = "Invalid configuration parameter"):
        super().__init__(f"Configuration error: {message}")

class OperatorError(QuantumSimError):
    """Exception raised for errors in quantum operator creation or manipulation."""
    def __init__(self, message: str = "Error in quantum operator operation"):
        super().__init__(f"Operator error: {message}")

class StatePreparationError(QuantumSimError):
    """Exception raised for errors in quantum state preparation."""
    def __init__(self, message: str = "Error in quantum state preparation"):
        super().__init__(f"State preparation error: {message}")

class HamiltonianError(QuantumSimError):
    """Exception raised for errors in Hamiltonian construction."""
    def __init__(self, message: str = "Error in Hamiltonian construction"):
        super().__init__(f"Hamiltonian error: {message}")

class SimulationError(QuantumSimError):
    """Base exception for simulation runtime errors."""
    def __init__(self, message: str = "Error during simulation execution"):
        super().__init__(f"Simulation error: {message}")

class TimeEvolutionError(SimulationError):
    """Exception raised for errors in time evolution calculations."""
    def __init__(self, message: str = "Error in time evolution"):
        super().__init__(f"Time evolution error: {message}")

class ConvergenceError(SimulationError):
    """Exception raised when numerical methods fail to converge."""
    def __init__(self, message: str = "Numerical method failed to converge"):
        super().__init__(f"Convergence error: {message}")

class MetricError(QuantumSimError):
    """Base exception for metric-related errors."""
    def __init__(self, message: str = "Error in metric operation"):
        super().__init__(f"Metric error: {message}")

class MetricCalculationError(MetricError):
    """Exception raised for errors in metric calculations."""
    def __init__(self, message: str = "Error calculating metric"):
        super().__init__(f"Metric calculation error: {message}")

class MetricVisualizationError(MetricError):
    """Exception raised for errors in metric visualization."""
    def __init__(self, message: str = "Error visualizing metric"):
        super().__init__(f"Metric visualization error: {message}")

# Decorator for consistent error handling
def error_handler(error_class):
    """Decorator for consistent error handling across the project.
    
    Args:
        error_class: The error class to use for the wrapped function
        
    Returns:
        Decorated function with error handling
        
    Example:
        @error_handler(SimulationError)
        def simulate():
            # Function implementation
            pass
    """
    def decorator(func):
        import functools
        import logging
        
        logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise error_class(f"Failed in {func.__name__}: {str(e)}") from e
        return wrapper
    return decorator