# -*- coding: utf-8 -*-
"""
Block-level components for E2Former neural network architecture.

This module re-exports block components for backward compatibility.
The actual implementations have been refactored into the blocks/ subfolder.
"""

# Re-export all block components from the subfolder
from .blocks import (
    CoefficientMapping,
    TransBlock,
    MessageBlock_escn,
    MessageBlock_eqv2,
    construct_radius_neighbor,
)

__all__ = [
    "CoefficientMapping",
    "TransBlock",
    "MessageBlock_escn", 
    "MessageBlock_eqv2",
    "construct_radius_neighbor",
]