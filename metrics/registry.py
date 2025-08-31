"""
Metric Registry Module
====================

This module provides the MetricRegistry class for managing available metric calculators.
"""

from typing import Dict, List, Type

from metrics.base import MetricCalculator
from metrics.mean_number import MeanNumberMetric
from metrics.sub_poissonian import SubPoissonianMetric
from metrics.wigner_negativity import WignerNegativityMetric
from metrics.coherence import CoherenceMetric

class MetricRegistry:
    """Registry for available metric calculators."""

    _metrics: Dict[str, Type[MetricCalculator]] = {
        'mean_num': MeanNumberMetric,
        'r_param': SubPoissonianMetric,
        'wigner_neg': WignerNegativityMetric,
        'coherence': CoherenceMetric
    }

    @classmethod
    def get_calculator(cls, metric_type: str) -> MetricCalculator:
        """Get metric calculator instance.

        Args:
            metric_type: Type of metric to calculate

        Returns:
            Appropriate metric calculator instance

        Raises:
            ValueError: If metric type is not supported
        """
        try:
            return cls._metrics[metric_type]()
        except KeyError:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    @classmethod
    def available_metrics(cls) -> List[str]:
        """Get list of available metrics.

        Returns:
            List of supported metric types
        """
        return list(cls._metrics.keys())
