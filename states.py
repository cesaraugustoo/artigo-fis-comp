"""
Quantum States Module
===================

This module provides classes and functions for creating and managing quantum states
used in the trapped ion-cavity quantum system simulation.

The states follow this ordering convention:
    1. Qubit (2-dimensional)
    2. Cavity field (b operators)
    3. Vibrational mode (a operators)

Classes:
    StateGenerator: Main class for creating and managing quantum states
    StateType: Enumeration of available initial state types
"""

from enum import Enum, auto
from typing import Dict, Optional, Union
import logging

from qutip import (Qobj, basis, coherent, coherent_dm, thermal_dm, ket2dm,
                  tensor)
import numpy as np

from operators import QuantumOperators
from exceptions import StatePreparationError

# Setup logging
logger = logging.getLogger(__name__)

class StateType(Enum):
    """Enumeration of available initial state types."""
    CLASSICAL = auto()  # |e⟩⟨e| ⊗ |β⟩⟨β| ⊗ ρth
    QUANTUM = auto()    # |+⟩⟨+| ⊗ |1⟩⟨1| ⊗ |2⟩⟨2|
    DRESSED_PLUS = auto()  # |+⟩⟨+| ⊗ |ψ⟩⟨ψ| ⊗ |φ⟩⟨φ|
    DRESSED_MINUS = auto() # |-⟩⟨-| ⊗ |ψ⟩⟨ψ| ⊗ |φ⟩⟨φ|
    CAT = auto()         # For Schrödinger cat states
    PURE_FOCK = auto() 
    
    @classmethod
    def from_str(cls, state_type: str) -> 'StateType':
        """Convert string to StateType.
        
        Args:
            state_type: String representation of state type
            
        Returns:
            Corresponding StateType enum value
            
        Raises:
            ValueError: If state_type is invalid
        """
        state_map = {
            'classical': cls.CLASSICAL,
            'quantum': cls.QUANTUM,
            'dressed_plus': cls.DRESSED_PLUS,
            'dressed_minus': cls.DRESSED_MINUS,
            'cat': cls.CAT,
            'pure_fock': cls.PURE_FOCK
        }
        try:
            return state_map[state_type.lower()]
        except KeyError:
            raise ValueError(f"Invalid state type: {state_type}. "
                           f"Valid options are: {list(state_map.keys())}")

class StateGenerator:
    """Generator class for quantum states in the trapped ion-cavity system."""
    
    def __init__(self, N_a: int, N_b: int):
        """Initialize the quantum state generator.
        
        Args:
            N_a: Dimension of vibrational mode Hilbert space
            N_b: Dimension of cavity field Hilbert space
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if N_a < 1 or N_b < 1:
            raise ValueError("Hilbert space dimensions must be positive integers")
            
        self.N_a = N_a
        self.N_b = N_b
        self.operators = QuantumOperators(N_a, N_b)
    
    def _generate_classical_state(self, params: Dict) -> Qobj:
        """Generate classical-like initial state:
        ρ = |e⟩⟨e| ⊗ |β⟩⟨β| ⊗ ρth
        
        Args:
            params: Dictionary containing state parameters
            
        Returns:
            Classical-like initial state
        """
        try:
            # Ensure basic operators are created
            self.operators.create_basic_operators()
            
            # Create excited state density matrix for qubit
            e = self.operators.get_operator('e')
            if e is None:
                raise StatePreparationError("Failed to get excited state operator")
            rho_qubit = ket2dm(e).to("CSR")
            
            # Create coherent state for cavity field
            rho_cavity = coherent_dm(self.N_b, params['beta']).to("CSR")
            
            # Create thermal state for vibrational mode
            rho_vibrational = thermal_dm(self.N_a, params['thermal_n']).to("CSR")
            
            # Create tensor product with correct ordering
            return tensor([rho_qubit, rho_cavity, rho_vibrational]).to("CSR")
            
        except Exception as e:
            logger.error(f"Failed to generate classical state: {str(e)}")
            raise StatePreparationError("Classical state preparation failed") from e
    
    def _generate_quantum_state(self, params: Dict) -> Qobj:
        """Generate quantum initial state:
        ρ = |+⟩⟨+| ⊗ |n⟩⟨n| ⊗ |m⟩⟨m|
        
        Args:
            params: Dictionary containing state parameters
            
        Returns:
            Quantum initial state
        """
        try:
            # Ensure basic operators are created
            self.operators.create_basic_operators()
            
            # Get qubit basis states
            e = self.operators.get_operator('e')
            g = self.operators.get_operator('g')
            if e is None or g is None:
                raise StatePreparationError("Failed to get qubit basis operators")
            
            # Create superposition state |+⟩ = (|g⟩ + |e⟩)/√2
            plus_state = ((g + e)/np.sqrt(2)).unit()
            rho_qubit = ket2dm(plus_state).to("CSR")
            
            # Create Fock state for cavity field
            rho_cavity = ket2dm(basis(self.N_b, params['cavity_n'])).to("CSR")
            
            # Create Fock state for vibrational mode
            rho_vibrational = ket2dm(basis(self.N_a, params['vib_n'])).to("CSR")
            
            # Create tensor product with correct ordering
            return tensor([rho_qubit, rho_cavity, rho_vibrational]).to("CSR")
            
        except Exception as e:
            logger.error(f"Failed to generate quantum state: {str(e)}")
            raise StatePreparationError("Quantum state preparation failed") from e
    
    def _generate_dressed_state(self, dressed_type: StateType, params: Dict) -> Qobj:
        """Generate dressed state configurations.
        
        Args:
            dressed_type: DRESSED_PLUS or DRESSED_MINUS
            params: Dictionary containing state parameters:
                - cavity_state: 'fock' or 'coherent'
                - cavity_param: parameter for cavity state
                - vib_state: 'fock' or 'thermal'
                - vib_param: parameter for vibrational state
                
        Returns:
            Dressed state configuration
            
        Raises:
            StatePreparationError: If state preparation fails
        """
        try:
            # Ensure basic operators are created
            self.operators.create_basic_operators()
            
            # Get qubit basis states
            e = self.operators.get_operator('e')
            g = self.operators.get_operator('g')
            if e is None or g is None:
                raise StatePreparationError("Failed to get qubit basis operators")
            
            # Create dressed state based on type
            if dressed_type == StateType.DRESSED_PLUS:
                qubit_state = ((e + g) / np.sqrt(2)).unit()
            else:  # DRESSED_MINUS
                qubit_state = ((e - g) / np.sqrt(2)).unit()
            
            rho_qubit = ket2dm(qubit_state).to("CSR")
            
            # Generate cavity state
            if params['cavity_state'] == 'fock':
                n = int(params['cavity_param'])
                rho_cavity = ket2dm(basis(self.N_b, n)).to("CSR")
            else:  # coherent
                alpha = complex(params['cavity_param'])
                rho_cavity = coherent_dm(self.N_b, alpha).to("CSR")
            
            # Generate vibrational state
            if params['vib_state'] == 'fock':
                m = int(params['vib_param'])
                rho_vibrational = ket2dm(basis(self.N_a, m)).to("CSR")
            else:  # thermal
                nth = float(params['vib_param'])
                rho_vibrational = thermal_dm(self.N_a, nth).to("CSR")
            
            # Create tensor product with correct ordering
            return tensor([rho_qubit, rho_cavity, rho_vibrational]).to("CSR")
            
        except Exception as e:
            logger.error(f"Failed to generate dressed state: {str(e)}")
            raise StatePreparationError("Dressed state preparation failed") from e

    def _generate_cat_state(self, params: Dict) -> Qobj:
        """Generate Schrödinger cat state configuration.
        
        Creates a state of the form:
        |ψ⟩ = N(|α⟩ ± |-α⟩) for the cavity field
        where N is the normalization factor.
        
        Args:
            params: Dictionary containing:
                - alpha: Complex amplitude for coherent states
                - parity: +1 for even cat, -1 for odd cat
                - qubit_state: 'e', 'g', '+', or '-'
                - vib_state: 'fock' or 'thermal'
                - vib_param: Parameter for vibrational state
                
        Returns:
            Cat state configuration
            
        Raises:
            StatePreparationError: If state preparation fails
        """
        try:
            # Ensure basic operators are created
            self.operators.create_basic_operators()
            
            # Get qubit basis states
            e = self.operators.get_operator('e')
            g = self.operators.get_operator('g')
            if e is None or g is None:
                raise StatePreparationError("Failed to get qubit basis operators")
            
            # Prepare qubit state
            qubit_state_map = {
                'e': e,
                'g': g,
                '+': ((e + g) / np.sqrt(2)).unit(),
                '-': ((e - g) / np.sqrt(2)).unit()
            }
            qubit_state = qubit_state_map.get(params['qubit_state'])
            if qubit_state is None:
                raise ValueError(f"Invalid qubit_state: {params['qubit_state']}")
            
            # Create cat state for cavity field
            alpha = complex(params['alpha'])
            parity = params['parity']  # +1 for even, -1 for odd
            
            # Generate superposition of coherent states
            coh_plus = coherent(self.N_b, alpha)
            coh_minus = coherent(self.N_b, -alpha)
            cat = (coh_plus + parity * coh_minus).unit()
            
            # Convert to density matrix
            rho_cavity = ket2dm(cat)
            
            # Generate vibrational state based on parameters
            if params['vib_state'] == 'fock':
                m = int(params['vib_param'])
                rho_vibrational = ket2dm(basis(self.N_a, m))
            else:  # thermal
                nth = float(params['vib_param'])
                rho_vibrational = thermal_dm(self.N_a, nth)
            
            # Create tensor product with correct ordering
            return tensor([
                ket2dm(qubit_state),
                rho_cavity,
                rho_vibrational
            ]).to("CSR")
            
        except Exception as e:
            logger.error(f"Failed to generate cat state: {str(e)}")
            raise StatePreparationError("Cat state preparation failed") from e
    
    def _generate_pure_fock_state(self, params: Dict) -> Qobj:
        """Generate pure Fock state configuration:
        ρ = |q⟩⟨q| ⊗ |n⟩⟨n| ⊗ |m⟩⟨m|
        where |q⟩ is either |e⟩ or |g⟩
        
        Args:
            params: Dictionary containing:
                - qubit_state: 'e' or 'g'
                - cavity_n: Cavity Fock number
                - vib_n: Vibrational Fock number
                
        Returns:
            Pure Fock state configuration
        """
        try:
            # Ensure basic operators are created
            self.operators.create_basic_operators()
            
            # Get qubit state
            if params['qubit_state'] == 'e':
                qubit_state = self.operators.get_operator('e')
            elif params['qubit_state'] == 'g':
                qubit_state = self.operators.get_operator('g')
            else:
                raise ValueError(f"Invalid qubit_state: {params['qubit_state']}")
            
            # Create pure states
            rho_qubit = ket2dm(qubit_state).to("CSR")
            rho_cavity = ket2dm(basis(self.N_b, params['cavity_n'])).to("CSR")
            rho_vibrational = ket2dm(basis(self.N_a, params['vib_n'])).to("CSR")
            
            # Create tensor product with correct ordering
            return tensor([rho_qubit, rho_cavity, rho_vibrational]).to("CSR")
            
        except Exception as e:
            logger.error(f"Failed to generate pure Fock state: {str(e)}")
            raise StatePreparationError("Pure Fock state preparation failed") from e
        
    def generate_initial_state(self, state_type: Union[str, StateType],
                             params: Optional[Dict] = None) -> Qobj:
        """Generate initial state for the simulation.
        
        Args:
            state_type: Type of initial state
            params: Optional parameters for state generation
            
        Returns:
            Initial quantum state density matrix
        """
        try:
            # Convert string to enum if necessary
            if isinstance(state_type, str):
                state_type = StateType.from_str(state_type)
            
            # Ensure we have parameters
            if params is None:
                params = self._get_default_params(state_type)
            
            # Generate appropriate state
            if state_type == StateType.CLASSICAL:
                return self._generate_classical_state(params)
            elif state_type == StateType.QUANTUM:
                return self._generate_quantum_state(params)
            elif state_type in [StateType.DRESSED_PLUS, StateType.DRESSED_MINUS]:
                return self._generate_dressed_state(state_type, params)
            elif state_type == StateType.CAT:
                return self._generate_cat_state(params)
            elif state_type == StateType.PURE_FOCK:
                return self._generate_pure_fock_state(params)
            else:
                raise ValueError(f"Unsupported state type: {state_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate initial state: {str(e)}")
            raise StatePreparationError("State preparation failed") from e
            
    def _get_default_params(self, state_type: StateType) -> Dict:
        """Get default parameters for state generation.
        
        Args:
            state_type: Type of initial state
            
        Returns:
            Dictionary of default parameters
        """
        if state_type == StateType.CLASSICAL:
            return {
                'beta': 1.0,  # Coherent state amplitude 
                'thermal_n': 2.0  # Thermal occupation number
            }
        elif state_type == StateType.QUANTUM:
            return {
                'cavity_n': 1,  # Cavity photon number
                'vib_n': 2  # Vibrational excitation number
            }
        elif state_type == StateType.CAT:
            return {
                'alpha': 1.0,            # Coherent state amplitude
                'parity': 1,             # Even cat state by default
                'qubit_state': 'e',      # Excited state by default
                'vib_state': 'fock',     # Fock state for vibration
                'vib_param': 0           # Ground state by default
            }
        elif state_type == StateType.PURE_FOCK:  # Add new case
            return {
                'qubit_state': 'e',      # Excited state by default
                'cavity_n': 0,           # Cavity ground state by default
                'vib_n': 0               # Vibrational ground state by default
            }
        elif state_type in [StateType.DRESSED_PLUS, StateType.DRESSED_MINUS]:
            return {
                'cavity_state': 'fock',  # 'fock' or 'coherent'
                'cavity_param': 1,       # Fock number or coherent amplitude
                'vib_state': 'fock',     # 'fock' or 'thermal'
                'vib_param': 0           # Fock number or thermal occupation
            }
   
    def partial_trace_qubit(self, state: Qobj) -> Qobj:
        """Calculate the reduced density matrix for the qubit.
        
        Args:
            state: Full system state
            
        Returns:
            Reduced density matrix for the qubit
        """
        return state.ptrace(0)
        
    def partial_trace_cavity(self, state: Qobj) -> Qobj:
        """Calculate the reduced density matrix for the cavity field.
        
        Args:
            state: Full system state
            
        Returns:
            Reduced density matrix for the cavity field
        """
        return state.ptrace(1)
        
    def partial_trace_vibrational(self, state: Qobj) -> Qobj:
        """Calculate the reduced density matrix for the vibrational mode.
        
        Args:
            state: Full system state
            
        Returns:
            Reduced density matrix for the vibrational mode
        """
        return state.ptrace(2)
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.operators.clear_cache()