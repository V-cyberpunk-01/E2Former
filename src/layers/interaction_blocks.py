# -*- coding: utf-8 -*-
"""
Interaction Blocks for E2Former.

This module re-exports interaction components for backward compatibility.
The actual implementations have been moved to blocks/interaction_blocks.py.
"""

# Re-export all interaction components from the blocks subfolder
from .blocks.interaction_blocks import (
    Body2_interaction,
    Body3_interaction_MACE,
)

__all__ = [
    "Body2_interaction",
    "Body3_interaction_MACE",
]