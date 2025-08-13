# -*- coding: utf-8 -*-
"""
E2Former Modular Implementation

This file serves as a compatibility layer, importing the main E2former and 
E2formerCluster classes from their respective modules.

The implementation has been split into:
- e2former_main.py: Core E2former model
- e2former_cluster.py: E2formerCluster model with hierarchical clustering

Author: E2Former Team
License: MIT
"""

# Import main classes from their separate modules
from .e2former_main import E2former, get_powers
from .e2former_cluster import E2formerCluster, E2AttentionArbOrder_sparse_forcluster

# Re-export for backward compatibility
__all__ = [
    "E2former",
    "E2formerCluster", 
    "E2AttentionArbOrder_sparse_forcluster",
    "get_powers",
]