"""
Quantum Operators Module
=======================

This module provides classes and functions for creating and managing quantum operators
used in the trapped ion-cavity quantum system simulation.

Classes:
    QuantumOperators: Main class for creating and managing quantum operators
    OperatorCache: Class for caching and managing operator instances

The operators follow this ordering convention:
    1. Qubit (2-dimensional)
    2. Cavity field (b operators)
    3. Vibrational mode (a operators)
"""

from typing import Dict, Optional, Tuple
import logging
import gc

from qutip import Qobj, basis, destroy, qeye, tensor
from scipy.sparse import csr_matrix

from exceptions import OperatorError

# Setup logging
logger = logging.getLogger(__name__)

class OperatorCache:
    """Cache manager for quantum operators to optimize memory usage."""
    
    def __init__(self):
        """Initialize an empty operator cache."""
        self._cache: Dict[str, Qobj] = {}
        
    def get(self, key: str) -> Optional[Qobj]:
        """Retrieve an operator from cache.
        
        Args:
            key: The identifier for the operator
            
        Returns:
            The cached operator or None if not found
        """
        return self._cache.get(key)
        
    def set(self, key: str, operator: Qobj) -> None:
        """Store an operator in cache.
        
        Args:
            key: The identifier for the operator
            operator: The quantum operator to cache
        """
        self._cache[key] = operator
        
    def clear(self) -> None:
        """Clear all cached operators and run garbage collection."""
        self._cache.clear()
        gc.collect()
        
    def __contains__(self, key: str) -> bool:
        """Check if an operator exists in cache."""
        return key in self._cache

class QuantumOperators:
    """Manager class for quantum operators in the trapped ion-cavity system."""
    
    def __init__(self, N_a: int, N_b: int):
        """Initialize the quantum operators manager.
        
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
        self._cache = OperatorCache()
        
    def create_basic_operators(self) -> Dict[str, Qobj]:
        """Create and cache basic quantum operators.
        
        Returns:
            Dictionary containing basic quantum operators
            
        Raises:
            OperatorError: If operator creation fails
        """
        try:
            # Create basic operators if not already cached
            if not self._cache._cache:
                # Qubit basis states
                self._cache.set('e', basis(2, 0))  # Excited state
                self._cache.set('g', basis(2, 1))  # Ground state
                
                # Field operators
                self._cache.set('b', destroy(self.N_b))  # Cavity field
                self._cache.set('a', destroy(self.N_a))  # Vibrational mode
                
                # Add frequently used combinations
                self._create_pauli_operators()
                self._create_identity_operators()
                
            return self._cache._cache
            
        except Exception as e:
            logger.error(f"Failed to create basic operators: {str(e)}")
            raise OperatorError("Basic operator creation failed") from e
            
    def _create_pauli_operators(self) -> None:
        """Create and cache Pauli operators."""
        e = self._cache.get('e')
        g = self._cache.get('g')
        
        # σ+ = |e⟩⟨g|
        self._cache.set('sp', (e * g.dag()).to("CSR"))
        # σ- = |g⟩⟨e|
        self._cache.set('sm', (g * e.dag()).to("CSR"))
        # σz = |e⟩⟨e| - |g⟩⟨g|
        self._cache.set('sz', (e * e.dag() - g * g.dag()).to("CSR"))
            
    def _create_identity_operators(self) -> None:
        """Create and cache identity operators for each subspace."""
        self._cache.set('Iq', qeye(2).to("CSR"))         # Qubit space
        self._cache.set('Ib', qeye(self.N_b).to("CSR"))  # Cavity field space
        self._cache.set('Ia', qeye(self.N_a).to("CSR"))  # Vibrational mode space
            
    def create_composite_operators(self) -> Dict[str, Qobj]:
        """Create composite quantum operators with correct ordering:
        qubit ⊗ cavity ⊗ vibrational
        
        Returns:
            Dictionary containing composite quantum operators
            
        Raises:
            OperatorError: If operator creation fails
        """
        try:
            # Ensure basic operators exist
            if not self._cache._cache:
                self.create_basic_operators()
                
            # Create composite operators if not already cached
            if 'sigma_z' not in self._cache:
                # Pauli operators tensor product with identities
                self._cache.set('sigma_z', 
                    tensor([self._cache.get('sz'),
                           self._cache.get('Ib'),
                           self._cache.get('Ia')]).to("CSR"))
                           
                self._cache.set('sigma_plus',
                    tensor([self._cache.get('sp'),
                           self._cache.get('Ib'),
                           self._cache.get('Ia')]).to("CSR"))
                           
                self._cache.set('sigma_minus',
                    tensor([self._cache.get('sm'),
                           self._cache.get('Ib'),
                           self._cache.get('Ia')]).to("CSR"))
                
                # Field operators tensor product with identities
                self._cache.set('b_op',
                    tensor([self._cache.get('Iq'),
                           self._cache.get('b'),
                           self._cache.get('Ia')]).to("CSR"))
                           
                self._cache.set('bd_op',
                    tensor([self._cache.get('Iq'),
                           self._cache.get('b').dag(),
                           self._cache.get('Ia')]).to("CSR"))
                           
                self._cache.set('a_op',
                    tensor([self._cache.get('Iq'),
                           self._cache.get('Ib'),
                           self._cache.get('a')]).to("CSR"))
                           
                self._cache.set('ad_op',
                    tensor([self._cache.get('Iq'),
                           self._cache.get('Ib'),
                           self._cache.get('a').dag()]).to("CSR"))
                
            return self._cache._cache
            
        except Exception as e:
            logger.error(f"Failed to create composite operators: {str(e)}")
            raise OperatorError("Composite operator creation failed") from e
            
    def get_operator(self, key: str) -> Optional[Qobj]:
        """Retrieve a specific operator from cache.
        
        Args:
            key: The identifier for the operator
            
        Returns:
            The requested quantum operator or None if not found
        """
        return self._cache.get(key)
        
    def clear_cache(self) -> None:
        """Clear all cached operators."""
        self._cache.clear()

def create_tensor_operator(op: Qobj, dims: Tuple[int, int, int], 
                         qubit_op: Optional[Qobj] = None) -> Qobj:
    """Create a tensor product operator for the system.
    
    Args:
        op: The operator to tensorize
        dims: Tuple of (qubit_dim, cavity_dim, vibrational_dim)
        qubit_op: Optional qubit operator (default: identity)
        
    Returns:
        Tensor product operator
        
    Raises:
        ValueError: If dimensions are invalid
    """
    if not all(d > 0 for d in dims):
        raise ValueError("All dimensions must be positive integers")
        
    if qubit_op is None:
        qubit_op = qeye(dims[0])
        
    return tensor([
        qubit_op,           # qubit dimension
        op,                 # target mode operator
        qeye(dims[2])      # other mode identity
    ]).to("CSR")