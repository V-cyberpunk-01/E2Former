# -*- coding: utf-8 -*-
"""
MACE blocks for E2Former.

This module re-exports MACE components for backward compatibility.
The actual implementations have been moved to blocks/maceblocks.py.
"""

# Re-export all MACE components from the blocks subfolder
from .blocks.maceblocks import (
    EquivariantProductBasisBlock,
    reshape_irrepstoe3nn,
)

__all__ = [
    "EquivariantProductBasisBlock",
    "reshape_irrepstoe3nn",
]