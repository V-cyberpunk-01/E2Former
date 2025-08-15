# -*- coding: utf-8 -*-
"""
SO2 rotation and convolution modules for E2Former.

This module re-exports SO2 components for backward compatibility.
The actual implementations have been moved to blocks/so2.py.
"""

# Re-export all SO2 components from the blocks subfolder
from .blocks.so2 import (
    CoefficientMappingModule,
    SO3_Rotation,
    SO2_Convolution,
    SO2_Convolution_sameorder,
    wigner_D,
    _init_edge_rot_mat,
)

__all__ = [
    "CoefficientMappingModule",
    "SO3_Rotation",
    "SO2_Convolution",
    "SO2_Convolution_sameorder",
    "wigner_D",
    "_init_edge_rot_mat",
]