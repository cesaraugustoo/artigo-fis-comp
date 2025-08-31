"""
Quantum Hamiltonians Module
==========================

This module provides classes and functions for creating and managing Hamiltonians
used in the trapped ion-cavity quantum system simulation.

The Hamiltonians follow the Lamb-Dicke regime (η ≪ 1) and use the Rotating Wave 
Approximation (RWA) when specified. Two types of Hamiltonians are supported:

H+ (Δ = ν):
    H = νa†a + ωb†b + (ω₀/2)σz + ḡ(σ₊ba + σ₋b†a†)

H- (Δ = -ν):
    H = νa†a + ωb†b + (ω₀/2)σz + ḡ(σ₊ba† + σ₋b†a)

Where ḡ ≡ ηg is the effective coupling constant.

Classes:
    HamiltonianType: Enumeration of available Hamiltonian types
    HamiltonianBuilder: Main class for creating system Hamiltonians
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple
import logging
from functools import lru_cache
import hashlib

from qutip import Qobj
import numpy as np

from exceptions import HamiltonianError
from operators import QuantumOperators

# Setup logging
logger = logging.getLogger(__name__)

class HamiltonianType(Enum):
    """Enumeration of available Hamiltonian types."""
    H_PLUS = auto()  # Δ = ν
    H_MINUS = auto() # Δ = -ν
    
    @classmethod
    def from_str(cls, ham_type: str) -> 'HamiltonianType':
        """Convert string to HamiltonianType.
        
        Args:
            ham_type: String representation of Hamiltonian type
            
        Returns:
            Corresponding HamiltonianType enum value
            
        Raises:
            ValueError: If ham_type is invalid
        """
        type_map = {
            'h_plus': cls.H_PLUS,
            'h_minus': cls.H_MINUS
        }
        try:
            return type_map[ham_type.lower()]
        except KeyError:
            raise ValueError(f"Invalid Hamiltonian type: {ham_type}")

def exp_plus(t: float, args: Dict) -> complex:
    """Time-dependent coefficient for positive rotating terms.
    
    Args:
        t: Time point
        args: Additional arguments (not used)
        
    Returns:
        Complex exponential e^(2iνt)
    """
    return np.exp(2.0j * t)

def exp_minus(t: float, args: Dict) -> complex:
    """Time-dependent coefficient for negative rotating terms.
    
    Args:
        t: Time point
        args: Additional arguments (not used)
        
    Returns:
        Complex exponential e^(-2iνt)
    """
    return np.exp(-2.0j * t)

class HamiltonianBuilder:
    """Builder class for quantum system Hamiltonians with caching."""
    
    def __init__(self, N_a: int, N_b: int, nu: float = 1.0, eta: float = 0.2):
        """Initialize the Hamiltonian builder.
        
        Args:
            N_a: Dimension of vibrational mode Hilbert space
            N_b: Dimension of cavity field Hilbert space
            nu: Trap frequency (default: 1.0)
            eta: Lamb-Dicke parameter (default: 0.2)
        
         Raises:
            ValueError: If dimensions or parameters are invalid
        """
        if N_a < 1 or N_b < 1:
            raise ValueError("Hilbert space dimensions must be positive integers")
        if nu <= 0:
            raise ValueError("Trap frequency must be positive")
        if eta <= 0:
            raise ValueError("Lamb-Dicke parameter must be positive")
            
        self.N_a = N_a
        self.N_b = N_b
        self.nu = nu
        self.eta = eta
        self.operators = QuantumOperators(N_a, N_b)
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=32)
    def _build_cached_hamiltonian(self, ham_type_str: str, g_value: float, use_rwa: bool,
                                nu: float, eta: float) -> Union[Qobj, List]:
        """Build and cache Hamiltonian.
        
        Args:
            ham_type_str: String representation of Hamiltonian type
            g_value: Coupling strength value
            use_rwa: Whether to use RWA
            nu: Trap frequency
            eta: Lamb-Dicke parameter
            
        Returns:
            Either a single Hamiltonian (RWA) or a list of 
            [static_terms, [time_dependent_terms]] (non-RWA)
        """
        # Convert string to enum
        ham_type = HamiltonianType.from_str(ham_type_str)
        
        # Ensure operators are created
        self.operators.create_composite_operators()
        
        # Calculate effective coupling
        g_eff = (eta * g_value) / nu
        
        # Build appropriate Hamiltonian
        if ham_type == HamiltonianType.H_PLUS:
            return self._build_h_plus(g_eff, use_rwa)
        else:  # H_MINUS
            return self._build_h_minus(g_eff, use_rwa)
    
    def build_hamiltonian(self, ham_type: Union[str, HamiltonianType], 
                        g_value: float, use_rwa: bool = False
                        ) -> Union[Qobj, List[Union[Qobj, List]]]:
        """Build the system Hamiltonian with caching.
        
        Args:
            ham_type: Type of Hamiltonian to build
            g_value: Coupling strength value
            use_rwa: Whether to use Rotating Wave Approximation
            
        Returns:
            Either a single Hamiltonian (RWA) or a list of 
            [static_terms, [time_dependent_terms]] (non-RWA)
            
        Raises:
            HamiltonianError: If Hamiltonian construction fails
        """
        try:
            # Convert to string if needed
            ham_type_str = ham_type.lower() if isinstance(ham_type, str) else ham_type.name.lower()
            
            # Perform RWA analysis
            analysis = self.analyze_rwa_validity(g_value)
            logger.debug(f"RWA analysis for g/ν = {analysis['g_nu_ratio']:.3f}:")
            logger.debug(f"Estimated error: {analysis['rwa_error_estimate']:.3e}")
            logger.debug(f"RWA recommended: {analysis['recommended_rwa']}")
            
            # Try to get from cache
            try:
                result = self._build_cached_hamiltonian(
                    ham_type_str, g_value, use_rwa, self.nu, self.eta)
                self._cache_hits += 1
                logger.debug(f"Cache hit for Hamiltonian {ham_type_str}, g={g_value}, RWA={use_rwa}")
            except Exception as e:
                self._cache_misses += 1
                logger.debug(f"Cache miss for Hamiltonian {ham_type_str}, g={g_value}, RWA={use_rwa}")
                raise HamiltonianError(f"Failed to build Hamiltonian: {str(e)}") from e
            
            # Track contributions after building
            contributions = self._track_term_contributions(g_value * self.eta, use_rwa)
            logger.debug("Hamiltonian term contributions:")
            for term, value in contributions.items():
                logger.debug(f"  {term}: {value:.3e}")
            
            # Apply optimizations
            result = self._optimize_hamiltonian(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build Hamiltonian: {str(e)}")
            raise HamiltonianError("Hamiltonian construction failed") from e
            
    def _build_h_plus(self, g_eff: float, use_rwa: bool) -> Union[Qobj, List]:
        """Build H+ Hamiltonian (Δ = ν).
        
        Args:
            g_eff: Effective coupling strength
            use_rwa: Whether to use RWA
            
        Returns:
            H+ Hamiltonian in appropriate form
        """
        # Get required operators
        sigma_plus = self.operators.get_operator('sigma_plus')
        sigma_minus = self.operators.get_operator('sigma_minus')
        b = self.operators.get_operator('b_op')
        bd = self.operators.get_operator('bd_op')
        a = self.operators.get_operator('a_op')
        ad = self.operators.get_operator('ad_op')
        
        if use_rwa:
            # RWA Hamiltonian: H = g(σ₊ba + σ₋b†a†)
            return g_eff * (sigma_plus * b * a +
                        sigma_minus * bd * ad)
        else:
            static_term1 = g_eff * (sigma_plus * b * a)
            static_term2 = g_eff * (sigma_minus * bd * ad)
            H_static = static_term1 + static_term2

            time_dependent_terms = [
                [g_eff * (sigma_plus * b * ad), exp_plus],
                [g_eff * (sigma_minus * bd * a), exp_minus]
            ]

            return [H_static] + time_dependent_terms
            
    def _build_h_minus(self, g_eff: float, use_rwa: bool) -> Union[Qobj, List]:
        """Build H- Hamiltonian (Δ = -ν).
        
        Args:
            g_eff: Effective coupling strength
            use_rwa: Whether to use RWA
            
        Returns:
            H- Hamiltonian in appropriate form
        """
        # Get required operators
        sigma_plus = self.operators.get_operator('sigma_plus')
        sigma_minus = self.operators.get_operator('sigma_minus')
        b = self.operators.get_operator('b_op')
        bd = self.operators.get_operator('bd_op')
        a = self.operators.get_operator('a_op')
        ad = self.operators.get_operator('ad_op')
        
        if use_rwa:
            # RWA Hamiltonian: H = g(σ₊ba† + σ₋b†a)
            return g_eff * (sigma_plus * b * ad + 
                          sigma_minus * bd * a)
        else:
            # Calculate the individual terms that are static in the interaction picture
            static_term1 = g_eff * (sigma_plus * b * ad)
            static_term2 = g_eff * (sigma_minus * bd * a)

            # Sum the static terms into a single Qobj (H0 for mesolve)
            H_static = static_term1 + static_term2
            
            time_dependent_terms = [
                [g_eff * (sigma_plus * b * a), exp_minus],
                [g_eff * (sigma_minus * bd * ad), exp_plus]
            ]
            
            # return static_terms + time_dependent_terms
            return [H_static] + time_dependent_terms
    
    def _track_term_contributions(self, g_eff: float, use_rwa: bool) -> Dict[str, float]:
        """Track relative contributions of Hamiltonian terms."""
        contributions = {
            'static': g_eff,
            'counter_rotating': 0.0 if use_rwa else g_eff,
            'g_nu_ratio': g_eff / self.nu
        }
        return contributions
        
    def _optimize_hamiltonian(self, H: Union[Qobj, List]) -> Union[Qobj, List]:
        """Apply performance optimizations to Hamiltonian.
        
        Converts operators to sparse CSR format while preserving the structure
        of the Hamiltonians:
        - RWA: single Qobj
        - Full: [static_terms_list, [H_td1, func1], [H_td2, func2]]
        """
        if isinstance(H, Qobj):
            return H.to("CSR")  # RWA case
        else:
            # Optimize the static part (first element)
            H_static_optimized = H[0].to("CSR")
            # Optimize the operator part of each time-dependent term
            td_terms_optimized = [[term[0].to("CSR"), term[1]] for term in H[1:]] 
            # Reconstruct the list in the correct format
            return [H_static_optimized] + td_terms_optimized
    
    def analyze_rwa_validity(self, g_value: float) -> Dict[str, float]:
        """Analyze validity of RWA for given coupling strength."""
        g_nu_ratio = g_value / self.nu
        return {
            'g_nu_ratio': g_nu_ratio,
            'rwa_error_estimate': g_nu_ratio**2 / 4,
            'recommended_rwa': g_nu_ratio < 0.1
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary containing cache hits and misses
        """
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': self._cache_hits + self._cache_misses
        }
        
    def clear_cache(self) -> None:
        """Clear Hamiltonian cache and operator cache."""
        self._build_cached_hamiltonian.cache_clear()
        self.operators.clear_cache()
        self._cache_hits = 0
        self._cache_misses = 0

    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()