"""
Quantum System Dissipation Module
===============================

This module provides functionality for including dissipative effects in the
quantum simulation through collapse operators. It handles both cavity decay
and qubit dephasing mechanisms.

Classes:
    CollapseOperatorBuilder: Builder class for system collapse operators
"""

from typing import List, Optional
import logging
import numpy as np

from qutip import Qobj, tensor, basis, sigmap, sigmam, sigmaz
from exceptions import OperatorError
from operators import QuantumOperators

# Setup logging
logger = logging.getLogger(__name__)

class CollapseOperatorBuilder:
    """Builder class for system collapse operators."""

    def __init__(self, N_a: int, N_b: int):
        """Initialize collapse operator builder.

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

    def build_collapse_operators(self, kappa: Optional[float] = None,
                               gamma: Optional[float] = None,
                               gamma_sp: Optional[float] = None) -> List[Qobj]:
        """Build system collapse operators.

        Args:
            kappa: Optional cavity decay rate (in units of ν)
            gamma: Optional qubit dephasing rate (in units of ν)
            gamma_sp: Optional atomic spontaneous decay rate (in units of ν)
                    Default value ≈ 0.0241 (corresponding to γ_sp/2π ≈ 11.5 MHz
                    with typical trap frequency ν/2π ≈ 476.2 kHz)

        Returns:
            List of collapse operators

        Raises:
            OperatorError: If operator creation fails
            ValueError: If rates are invalid
        """
        try:
            c_ops = []

            # Validate rates if provided
            if kappa is not None and kappa < 0:
                raise ValueError("Cavity decay rate must be non-negative")
            if gamma is not None and gamma < 0:
                raise ValueError("Qubit dephasing rate must be non-negative")
            if gamma_sp is not None and gamma_sp < 0:
                raise ValueError("Atomic spontaneous decay rate must be non-negative")

            # Ensure basic operators are created
            self.operators.create_basic_operators()

            # Add cavity decay operator if rate provided
            if kappa is not None and kappa > 0:
                cavity_decay = self._build_cavity_decay(kappa)
                c_ops.append(cavity_decay)
                logger.debug(f"Added cavity decay operator with rate κ = {kappa}")

            # Add qubit dephasing operator if rate provided
            if gamma is not None and gamma > 0:
                qubit_dephasing = self._build_qubit_dephasing(gamma)
                c_ops.append(qubit_dephasing)
                logger.debug(f"Added qubit dephasing operator with rate γ = {gamma}")

            # Add spontaneous decay operator if rate provided
            if gamma_sp is not None and gamma_sp > 0:
                spontaneous_decay = self._build_spontaneous_decay(gamma_sp)
                c_ops.append(spontaneous_decay)
                logger.debug(f"Added spontaneous decay operator with rate Γ = {gamma_sp}")

            return c_ops

        except Exception as e:
            logger.error(f"Failed to build collapse operators: {str(e)}")
            raise OperatorError("Collapse operator creation failed") from e

    def _build_cavity_decay(self, kappa: float) -> Qobj:
        """Build cavity decay collapse operator.

        The cavity decay operator is √κ b ⊗ I, where b is the cavity
        lowering operator and I is the identity for other subsystems.

        Args:
            kappa: Cavity decay rate

        Returns:
            Cavity decay operator
        """
        # Get cavity lowering operator and identities
        b = self.operators.get_operator('b')  # Cavity lowering operator
        Iq = self.operators.get_operator('Iq')  # Qubit identity
        Ia = self.operators.get_operator('Ia')  # Vibrational mode identity

        # Create tensor product with correct ordering:
        # qubit ⊗ cavity ⊗ vibrational
        return np.sqrt(kappa) * tensor([
            Iq,          # qubit identity
            b,           # cavity lowering operator
            Ia           # vibrational identity
        ]).to("CSR")

    def _build_qubit_dephasing(self, gamma: float) -> Qobj:
        """Build qubit dephasing collapse operator.

        The qubit dephasing operator is √γ σz ⊗ I, where σz is the Pauli Z
        operator and I is the identity for other subsystems.

        Args:
            gamma: Qubit dephasing rate

        Returns:
            Qubit dephasing operator
        """
        # Get Pauli Z operator and identities
        sz = self.operators.get_operator('sz')  # Pauli Z operator
        Ib = self.operators.get_operator('Ib')  # Cavity identity
        Ia = self.operators.get_operator('Ia')  # Vibrational identity

        # Create tensor product with correct ordering:
        # qubit ⊗ cavity ⊗ vibrational
        return np.sqrt(gamma) * tensor([
            sz,          # Pauli Z operator
            Ib,         # cavity identity
            Ia          # vibrational identity
        ]).to("CSR")

    def _build_spontaneous_decay(self, gamma_sp: float) -> Qobj:
        """Build atomic spontaneous decay collapse operator.

        The spontaneous decay operator is √Γ σ₋ ⊗ I, where σ₋ is the
        qubit lowering operator and I is the identity for other subsystems.

        Args:
            gamma_sp: Atomic spontaneous decay rate

        Returns:
            Spontaneous decay operator
        """
        # Get lowering operator and identities
        sm = self.operators.get_operator('sm')  # Lowering operator
        Ib = self.operators.get_operator('Ib')  # Cavity identity
        Ia = self.operators.get_operator('Ia')  # Vibrational identity

        # Create tensor product with correct ordering:
        # qubit ⊗ cavity ⊗ vibrational
        return np.sqrt(gamma_sp) * tensor([
            sm,          # Lowering operator
            Ib,         # cavity identity
            Ia          # vibrational identity
        ]).to("CSR")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.operators.clear_cache()