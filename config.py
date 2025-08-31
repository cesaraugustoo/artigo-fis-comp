"""
Configuration Module
==================

This module provides configuration classes and validation for the quantum simulation
project. It centralizes all configuration parameters and ensures their consistency
across the project.

Classes:
    BaseConfig: Base configuration class with validation
    SimulationConfig: Main simulation configuration
    MetricsConfig: Configuration for metrics calculation and plotting
    PlotStyle: Configuration for plot styling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import numpy as np
import yaml

from exceptions import ConfigurationError

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    """Base configuration class with validation capabilities."""

    def validate(self) -> bool:
        """Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            self._validate()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ConfigurationError(str(e))

    def _validate(self) -> None:
        """Implementation-specific validation logic.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Validation not implemented")

@dataclass
class PlotStyle(BaseConfig):
    """Configuration for plot styling."""
    figsize: tuple = (12, 8)
    dpi: int = 300
    grid: bool = True
    style: str = 'seaborn'

    def _validate(self) -> None:
        """Validate plot style configuration."""
        if len(self.figsize) != 2:
            raise ConfigurationError("figsize must be a tuple of (width, height)")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ConfigurationError("dpi must be a positive integer")
        if not isinstance(self.grid, bool):
            raise ConfigurationError("grid must be a boolean")

@dataclass
class MetricsConfig(BaseConfig):
    """Configuration for metrics calculation and plotting."""
    metric_type: str
    output_dir: Path
    plot: bool = False
    plot_style: PlotStyle = field(default_factory=PlotStyle)
    use_gt_scale: bool = False
    sub_poiss_only: bool = False
    num_analysis: bool = False  # New field for numerical analysis

    # Wigner function visualization options
    add_contours: bool = True  # Add contour lines to 2D Wigner plots
    apply_smoothing: bool = True  # Apply Gaussian smoothing to Wigner plots
    use_consistent_scale: bool = True  # Use consistent color scale across plots

    # Parallelization options
    use_parallel: bool = True  # Use parallel processing for computationally intensive metrics
    max_workers: Optional[int] = None  # Number of worker processes (None = auto-detect)

    # Numerical tolerance options
    numerical_tolerances: Dict[str, float] = field(default_factory=lambda: {
        'truncation': 1e-12,     # Threshold for small values to set to zero
        'integration': 1e-10,    # Integration precision threshold
        'normalization': 1e-8,   # Allowed error in normalization
        'negativity_floor': 1e-9 # Threshold below which negative values are considered numerical artifacts
    })

    def _validate(self) -> None:
        """Validate metrics configuration."""
        if not self.metric_type:
            raise ConfigurationError("metric_type must be specified")
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Validate boolean flags
        for flag_name in ['plot', 'use_gt_scale', 'sub_poiss_only', 'num_analysis',
                          'add_contours', 'apply_smoothing', 'use_consistent_scale', 'use_parallel']:
            if not isinstance(getattr(self, flag_name), bool):
                raise ConfigurationError(f"{flag_name} must be a boolean")

        # Validate max_workers
        if self.max_workers is not None and (not isinstance(self.max_workers, int) or self.max_workers < 1):
            raise ConfigurationError("max_workers must be a positive integer or None")

        # Validate numerical tolerances
        if not isinstance(self.numerical_tolerances, dict):
            raise ConfigurationError("numerical_tolerances must be a dictionary")
        for key, value in self.numerical_tolerances.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigurationError(f"Numerical tolerance '{key}' must be a non-negative number")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate plot style if plotting is enabled
        if self.plot:
            if isinstance(self.plot_style, dict):
                self.plot_style = PlotStyle(**self.plot_style)
            self.plot_style.validate()

@dataclass
class SimulationConfig(BaseConfig):
    """Main configuration for quantum simulation."""
    # Hilbert space dimensions
    N_a: int  # Vibrational mode dimension
    N_b: int  # Cavity field dimension

    # System parameters
    hamiltonian: str  # 'h_plus' or 'h_minus'
    initial_state: str  # 'classical', 'quantum', 'dressed_plus', or 'dressed_minus'
    nu: float = 1.0  # Trap frequency
    eta: float = 0.2  # Lamb-Dicke parameter

    # Classical state parameters
    beta: float = 1.0  # Coherent state amplitude
    thermal_n: float = 2.0  # Thermal occupation number

    # Quantum and Fock state parameters
    cavity_n: int = 1  # Cavity photon number
    vib_n: int = 2  # Vibrational excitation number

    # Dressed state parameters
    cavity_state: Optional[str] = None  # 'fock' or 'coherent'
    cavity_param: Optional[float] = None  # parameter for cavity state
    vib_state: Optional[str] = None  # 'fock' or 'thermal'
    vib_param: Optional[float] = None  # parameter for vibrational state

    # Cat state parameters
    alpha: Optional[complex] = None  # Coherent state amplitude for cat states
    cat_parity: Optional[int] = None  # +1 for even, -1 for odd cat states
    qubit_state: Optional[str] = None  # 'e', 'g', '+', or '-' (also used in the Fock state)

    # Loss parameters (optional)
    kappa: Optional[float] = None  # Cavity decay rate (in units of ν)
    gamma: Optional[float] = None  # Qubit dephasing rate (in units of ν)
    gamma_sp: Optional[float] = None  # Atomic spontaneous decay rate (in units of ν)

    # Evolution parameters
    t_max: float = 40.0  # Maximum time in chosen units
    use_gt_scale: bool = False  # Whether to use gt scaling instead of νt
    n_steps: int = 200  # Number of time steps
    g_values: List[float] = field(default_factory=lambda: [0.2, 2.0])

    # Simulation options
    rwa: bool = False  # Use Rotating Wave Approximation
    metric: Optional[str] = None  # Optional metric to calculate
    plot: bool = False  # Generate plots for metric results
    sub_poiss: bool = False  # Show only sub-Poissonian statistics (R ≥ 0)
    num_analysis: bool = False  # Enable numerical analysis

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path('results'))
    data_dir: Optional[Path] = None  # Optional directory to load existing data from

    # Debug options
    profile: bool = False  # Enable profiling
    debug: bool = False  # Enable debug output

    def get_state_params(self) -> Dict:
        """Get state-specific parameters based on initial state type.

        Returns:
            Dictionary of parameters for state generation
        """
        if self.initial_state == 'classical':
            return {
                'beta': self.beta,
                'thermal_n': self.thermal_n
            }
        elif self.initial_state == 'quantum':
            return {
                'cavity_n': self.cavity_n,
                'vib_n': self.vib_n
            }
        elif self.initial_state == 'pure_fock':
            return {
                'qubit_state': self.qubit_state,
                'cavity_n': self.cavity_n,
                'vib_n': self.vib_n
            }
        elif self.initial_state in ['dressed_plus', 'dressed_minus']:
            return {
                'cavity_state': self.cavity_state,
                'cavity_param': self.cavity_param,
                'vib_state': self.vib_state,
                'vib_param': self.vib_param
            }
        elif self.initial_state == 'cat':
            return {
                'alpha': self.alpha,
                'parity': self.cat_parity,
                'qubit_state': self.qubit_state,
                'vib_state': self.vib_state,
                'vib_param': self.vib_param
            }
        return {}

    def _validate(self) -> None:
        """Validate simulation configuration."""
        # Validate Hilbert space dimensions
        if self.N_a < 1 or self.N_b < 1:
            raise ConfigurationError("Hilbert space dimensions must be positive integers")

        # Validate Hamiltonian type
        if self.hamiltonian not in ['h_plus', 'h_minus']:
            raise ConfigurationError("Hamiltonian must be either 'h_plus' or 'h_minus'")

        # Validate initial state type
        valid_states = ['classical', 'quantum', 'dressed_plus', 'dressed_minus', 'cat', 'pure_fock']
        if self.initial_state not in valid_states:
            raise ConfigurationError(
                f"Initial state must be one of: {', '.join(valid_states)}")

        # Validate state-specific parameters
        if self.initial_state == 'classical':
            if self.beta < 0:
                raise ConfigurationError("Coherent state amplitude must be non-negative")
            if self.thermal_n < 0:
                raise ConfigurationError("Thermal occupation number must be non-negative")

        elif self.initial_state == 'quantum':
            if self.cavity_n < 0:
                raise ConfigurationError("Cavity photon number must be non-negative")
            if self.vib_n < 0:
                raise ConfigurationError("Vibrational excitation number must be non-negative")

        elif self.initial_state in ['dressed_plus', 'dressed_minus']:
            if self.cavity_state is not None and self.cavity_state not in ['fock', 'coherent']:
                raise ConfigurationError("Cavity state must be either 'fock' or 'coherent'")
            if self.vib_state is not None and self.vib_state not in ['fock', 'thermal']:
                raise ConfigurationError("Vibrational state must be either 'fock' or 'thermal'")

            # Validate parameters if states are specified
            if self.cavity_state == 'fock' and self.cavity_param is not None:
                if not float(self.cavity_param).is_integer() or self.cavity_param < 0:
                    raise ConfigurationError("Fock state parameter must be a non-negative integer")
            if self.vib_state == 'fock' and self.vib_param is not None:
                if not float(self.vib_param).is_integer() or self.vib_param < 0:
                    raise ConfigurationError("Fock state parameter must be a non-negative integer")
            if self.vib_state == 'thermal' and self.vib_param is not None and self.vib_param < 0:
                raise ConfigurationError("Thermal state parameter must be non-negative")

        elif self.initial_state == 'cat':
            if self.cat_parity not in [1, -1]:
                raise ConfigurationError("Cat state parity must be either +1 (even) or -1 (odd)")
            if self.qubit_state not in ['e', 'g', '+', '-']:
                raise ConfigurationError("Qubit state must be one of: e, g, +, -")
            if self.vib_state not in ['fock', 'thermal']:
                raise ConfigurationError("Vibrational state must be either 'fock' or 'thermal'")
            if self.vib_state == 'fock':
                if self.vib_param is not None and (not float(self.vib_param).is_integer() or self.vib_param < 0):
                    raise ConfigurationError("Fock state parameter must be a non-negative integer")
            if self.vib_state == 'thermal' and self.vib_param is not None and self.vib_param < 0:
                raise ConfigurationError("Thermal state parameter must be non-negative")

        elif self.initial_state == 'pure_fock':
            if self.qubit_state not in ['e', 'g']:
                raise ConfigurationError("Qubit state must be either 'e' or 'g'")
            if self.cavity_n < 0:
                raise ConfigurationError("Cavity photon number must be non-negative")
            if self.vib_n < 0:
                raise ConfigurationError("Vibrational excitation number must be non-negative")

        # Validate physical parameters
        if self.nu <= 0:
            raise ConfigurationError("Trap frequency must be positive")
        if self.eta <= 0:
            raise ConfigurationError("Lamb-Dicke parameter must be positive")

        # Validate loss parameters if provided
        if self.kappa is not None and self.kappa < 0:
            raise ConfigurationError("Cavity decay rate must be non-negative")
        if self.gamma is not None and self.gamma < 0:
            raise ConfigurationError("Qubit dephasing rate must be non-negative")
        if self.gamma_sp is not None and self.gamma_sp < 0:
            raise ConfigurationError("Atomic spontaneous decay rate must be non-negative")

        # Validate evolution parameters
        if self.t_max <= 0:
            raise ConfigurationError("Maximum time must be positive")
        if self.n_steps < 1:
            raise ConfigurationError("Number of time steps must be positive")
        if not isinstance(self.use_gt_scale, bool):
            raise ConfigurationError("use_gt_scale must be a boolean")

        if not self.g_values:
            raise ConfigurationError("Must provide at least one coupling strength")
        if any(g <= 0 for g in self.g_values):
            raise ConfigurationError("All coupling strengths must be positive")

        # Validate output directory
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def solver_options(self) -> Dict:
        """Get solver options for quantum evolution.

        Returns:
            Dictionary of solver options
        """
        return {
            'atol': 1e-8,          # Absolute tolerance
            'rtol': 1e-6,          # Relative tolerance
            'nsteps': 2500,        # Maximum internal steps
            'method': 'adams',     # Integration method
            'progress_bar': 'tqdm' # Progress bar type
        }

    def get_evolution_times(self, g_value: float) -> tuple:
        """Get time points for evolution at given coupling strength.

        Args:
            g_value: Coupling strength value

        Returns:
            Tuple of (times, scaled_times)
        """
        if self.use_gt_scale:
            # gt scaling
            abs_t_max = self.t_max / g_value
            times = np.linspace(0, abs_t_max, self.n_steps)
            scaled_times = g_value * times  # In gt units
        else:
            # νt scaling (ν = 1 in our units)
            times = np.linspace(0, self.t_max, self.n_steps)
            scaled_times = times  # In νt units

        return times, scaled_times

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            'N_a': self.N_a,
            'N_b': self.N_b,
            'hamiltonian': self.hamiltonian,
            'initial_state': self.initial_state,
            'nu': self.nu,
            'eta': self.eta,
            'kappa': self.kappa,
            'gamma': self.gamma,
            'gamma_sp': self.gamma_sp,
            't_max': self.t_max,
            'use_gt_scale': self.use_gt_scale,
            'n_steps': self.n_steps,
            'g_values': self.g_values,
            'rwa': self.rwa,
            'metric': self.metric,
            'plot': self.plot,
            'sub_poiss': self.sub_poiss,
            'output_dir': str(self.output_dir),
            'data_dir': str(self.data_dir) if self.data_dir else None,
            'profile': self.profile,
            'debug': self.debug
        }

        # Add state-specific parameters
        if self.initial_state == 'classical':
            config_dict.update({
                'beta': self.beta,
                'thermal_n': self.thermal_n
            })
        elif self.initial_state == 'quantum':
            config_dict.update({
                'cavity_n': self.cavity_n,
                'vib_n': self.vib_n
            })
        elif self.initial_state == 'pure_fock':
            config_dict.update({
                'qubit_state': self.qubit_state,
                'cavity_n': self.cavity_n,
                'vib_n': self.vib_n
            })
        elif self.initial_state in ['dressed_plus', 'dressed_minus']:
            config_dict.update({
                'cavity_state': self.cavity_state,
                'cavity_param': self.cavity_param,
                'vib_state': self.vib_state,
                'vib_param': self.vib_param
            })
        elif self.initial_state == 'cat':
            config_dict.update({
                'alpha': self.alpha,
                'cat_parity': self.cat_parity,
                'qubit_state': self.qubit_state,
                'vib_state': self.vib_state,
                'vib_param': self.vib_param
            })

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulationConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns:
            SimulationConfig instance
        """
        # Convert output_dir and data_dir strings to Path objects
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        if 'data_dir' in config_dict and config_dict['data_dir'] is not None:
            config_dict['data_dir'] = Path(config_dict['data_dir'])
        return cls(**config_dict)

    def to_yaml(self, filepath: Path) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Path to save the YAML file

        Raises:
            ConfigurationError: If saving fails
        """
        try:
            # Convert to dictionary
            config_dict = self.to_dict()

            # Custom representer for complex numbers
            def complex_representer(dumper, data):
                return dumper.represent_scalar('!complex', f'{data.real}+{data.imag}j')

            # Register the representer
            yaml.add_representer(complex, complex_representer)

            # Write to file
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration to YAML: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}") from e

    @classmethod
    def from_yaml(cls, filepath: Path) -> 'SimulationConfig':
        """Load configuration from YAML file.

        Args:
            filepath: Path to the YAML file

        Returns:
            SimulationConfig instance

        Raises:
            ConfigurationError: If loading fails
        """
        try:
            # Custom constructor for complex numbers
            def complex_constructor(loader, node):
                value = loader.construct_scalar(node)
                return complex(value)

            # Register the constructor
            yaml.add_constructor('!complex', complex_constructor)

            # Read from file
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)

            return cls.from_dict(config_dict)

        except Exception as e:
            logger.error(f"Failed to load configuration from YAML: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e