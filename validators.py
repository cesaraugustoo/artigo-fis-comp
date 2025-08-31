"""
State Validation Module
=====================

This module provides validation utilities for quantum states and evolution results
in the trapped ion-cavity quantum system simulation.

Classes:
    StateValidationResult: Container for validation results
    StateValidator: Main validator class for quantum states
"""

from typing import List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field

from qutip import Qobj
import numpy as np

from exceptions import StatePreparationError, SimulationError

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class StateValidationResult:
    """Results of state validation."""
    is_valid: bool
    message: str = ""
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of validation result."""
        result = f"Valid: {self.is_valid}\nMessage: {self.message}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

class StateValidator:
    """Validator class for quantum states."""
    
    @staticmethod
    def validate_state_lists(states_full: List[Qobj], 
                           states_rwa: Optional[List[Qobj]] = None,
                           check_dims: bool = True) -> StateValidationResult:
        """Validate lists of quantum states.
        
        Args:
            states_full: List of states from full Hamiltonian evolution
            states_rwa: Optional list of states from RWA evolution
            check_dims: Whether to check state dimensions
            
        Returns:
            StateValidationResult containing validation results
        """
        try:
            # Basic validation
            if not states_full:
                return StateValidationResult(
                    False, 
                    "No states provided for full evolution"
                )
            
            # Check if both lists are provided
            if states_rwa is not None:
                if len(states_full) != len(states_rwa):
                    return StateValidationResult(
                        False,
                        "Full and RWA state lists must have same length",
                        {'full_len': len(states_full), 'rwa_len': len(states_rwa)}
                    )
            
            # Validate individual states
            for idx, state in enumerate(states_full):
                if not isinstance(state, Qobj):
                    return StateValidationResult(
                        False,
                        f"Invalid state object at index {idx} in full evolution",
                        {'index': idx, 'type': type(state)}
                    )
                    
                if states_rwa and not isinstance(states_rwa[idx], Qobj):
                    return StateValidationResult(
                        False,
                        f"Invalid state object at index {idx} in RWA evolution",
                        {'index': idx, 'type': type(states_rwa[idx])}
                    )
            
            # Check dimensions if requested
            if check_dims and states_full:
                dims = states_full[0].dims
                for idx, state in enumerate(states_full[1:], 1):
                    if state.dims != dims:
                        return StateValidationResult(
                            False,
                            f"Inconsistent dimensions at index {idx} in full evolution",
                            {'expected': dims, 'found': state.dims}
                        )
                        
                if states_rwa:
                    for idx, state in enumerate(states_rwa):
                        if state.dims != dims:
                            return StateValidationResult(
                                False,
                                f"Inconsistent dimensions at index {idx} in RWA evolution",
                                {'expected': dims, 'found': state.dims}
                            )
            
            return StateValidationResult(True, "All states validated successfully")
            
        except Exception as e:
            logger.error(f"State validation failed: {str(e)}")
            raise ValueError(f"State validation failed: {str(e)}")
    
    @staticmethod
    def validate_state_evolution(times: np.ndarray, states: List[Qobj]
                               ) -> StateValidationResult:
        """Validate state evolution results.
        
        Args:
            times: Array of time points
            states: List of evolved states
            
        Returns:
            StateValidationResult containing validation results
        """
        try:
            # Check time array
            if not isinstance(times, np.ndarray):
                return StateValidationResult(
                    False,
                    "Time points must be a numpy array",
                    {'type': type(times)}
                )
            
            # Check matching lengths
            if len(times) != len(states):
                return StateValidationResult(
                    False,
                    "Number of time points must match number of states",
                    {'times_len': len(times), 'states_len': len(states)}
                )
            
            # Validate time ordering
            if not np.all(np.diff(times) > 0):
                return StateValidationResult(
                    False,
                    "Time points must be strictly increasing"
                )
            
            # Basic state validation
            basic_validation = StateValidator.validate_state_lists(states)
            if not basic_validation.is_valid:
                return basic_validation
            
            return StateValidationResult(True, "Evolution results validated successfully")
            
        except Exception as e:
            logger.error(f"Evolution validation failed: {str(e)}")
            raise ValueError(f"Evolution validation failed: {str(e)}")
            
    @staticmethod
    def validate_and_prepare_states(states_full: List[Qobj],
                                  states_rwa: Optional[List[Qobj]] = None,
                                  times: Optional[np.ndarray] = None
                                  ) -> Tuple[List[Qobj], Optional[List[Qobj]]]:
        """Validate and prepare states for simulation or analysis.
        
        Args:
            states_full: List of states from full Hamiltonian evolution
            states_rwa: Optional list of states from RWA evolution
            times: Optional array of time points
            
        Returns:
            Tuple of validated (and possibly prepared) state lists
            
        Raises:
            StatePreparationError: If validation or preparation fails
        """
        try:
            # Validate states
            validation_result = StateValidator.validate_state_lists(
                states_full, states_rwa)
            
            if not validation_result.is_valid:
                raise StatePreparationError(validation_result.message)
            
            # Validate evolution if times are provided
            if times is not None:
                evolution_result = StateValidator.validate_state_evolution(
                    times, states_full)
                
                if not evolution_result.is_valid:
                    raise StatePreparationError(evolution_result.message)
            
            return states_full, states_rwa
            
        except Exception as e:
            logger.error(f"State preparation failed: {str(e)}")
            raise StatePreparationError(f"State preparation failed: {str(e)}")