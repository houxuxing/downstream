"""
3D Deep Learning Models for Medical Image Analysis

This package provides utilities for the AnatCL downstream tasks.
"""

# Classifier (用于下游分类任务)
from .classifier import AnatCLClassifier

# Freezing utilities (参数冻结工具)
from .freezing import (
    setup_parameter_freezing,
    set_frozen_batchnorm_eval,
)

__all__ = [
    "AnatCLClassifier",
    "setup_parameter_freezing",
    "set_frozen_batchnorm_eval",
]
