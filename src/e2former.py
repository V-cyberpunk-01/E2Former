# -*- coding: utf-8 -*-
"""
E2Former - Equivariant Transformer for Molecular Property Prediction

This file now serves as a compatibility layer, importing all components from the modularized files.
The original monolithic implementation has been split into:
- e2former_utils.py: Configuration and utilities
- base_modules.py: Base classes and helper functions
- interaction_blocks.py: Two-body and three-body interactions
- attention.py: Attention mechanisms
- embeddings.py: Embedding networks
- blocks.py: Transformer blocks
- e2former_modular.py: Main model classes
"""

# Import all components from modular files for backward compatibility

# Configuration and utilities
from .e2former_utils import (
    # Constants
    _AVG_DEGREE,
    _USE_BIAS,
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_DROPOUT_RATE,
    WEIGHT_INIT_RANGE,
    EMBEDDING_INIT_RANGE,
    EPSILON,
    CACHE_SIZE,
    # Configuration classes
    AttentionConfig,
    NetworkConfig,
    TrainingConfig,
    E2FormerConfig,
    # Utility functions
    construct_radius_neighbor,
    get_irreps_from_config,
    calculate_fan_in,
    init_weights,
    get_activation_fn,
    safe_divide,
    compute_degree,
    segment_mean,
    create_edge_mask,
)

# Base modules
from .base_modules import (
    BaseEquivariantModule,
    cached_spherical_harmonics,
    init_embeddings,
    irreps_times,
    no_weight_decay,
    create_trans_block,
)

# Interaction blocks
from .interaction_blocks import (
    Body2_interaction,
    Body3_interaction_MACE,
)

# Attention mechanisms
from .attention import (
    E2AttentionArbOrder_sparse,
    E2AttentionArbOrder_sparse_forcluster,
)

# Embedding networks
from .embeddings import (
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_higherorder_v3,
    EdgeDegreeEmbeddingNetwork_eqv2,
    BOOEmbedding,
    CoefficientMapping,
)

# Transformer blocks
from .blocks import (
    TransBlock,
    MessageBlock_escn,
    MessageBlock_eqv2,
)

# Main model classes
from .e2former_modular import (
    E2former,
    E2formerCluster,
    get_powers,
)

# For compatibility, also expose commonly used imports
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from sklearn.cluster import KMeans
from torch import logical_not, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_scatter import scatter_mean

# FairChem imports
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2
from fairchem.core.models.escn.escn import SO2Block
from fairchem.core.models.escn.so3 import SO3_Embedding, SO3_Rotation
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

# Module utilities (re-export commonly used ones)
from .module_utils import (
    DropPath_BL,
    Electron_Density_Descriptor,
    EquivariantDropout,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    FeedForwardNetwork_escn,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
    GaussianLayer_Edgetype,
    GaussianRadialBasisLayer,
    GaussianSmearing,
    Learn_PolynomialDistance,
    RadialFunction,
    RadialProfile,
    SmoothLeakyReLU,
    SO3_Grid,
    SO3_Linear2Scalar_e2former,
    SO3_Linear_e2former,
    get_normalization_layer,
    polynomial,
)

from .so2 import _init_edge_rot_mat
from .wigner6j.tensor_product import (
    E2TensorProductArbitraryOrder,
    Simple_TensorProduct_oTchannel,
)

from .maceblocks import EquivariantProductBasisBlock, reshape_irrepstoe3nn

# Define __all__ for clean imports
__all__ = [
    # Main classes
    "E2former",
    "E2formerCluster",
    # Attention
    "E2AttentionArbOrder_sparse",
    "E2AttentionArbOrder_sparse_forcluster",
    # Embeddings
    "EdgeDegreeEmbeddingNetwork_higherorder",
    "EdgeDegreeEmbeddingNetwork_higherorder_v3",
    "EdgeDegreeEmbeddingNetwork_eqv2",
    "BOOEmbedding",
    "CoefficientMapping",
    # Blocks
    "TransBlock",
    "MessageBlock_escn",
    "MessageBlock_eqv2",
    # Interactions
    "Body2_interaction",
    "Body3_interaction_MACE",
    # Base modules
    "BaseEquivariantModule",
    # Configuration
    "AttentionConfig",
    "NetworkConfig",
    "TrainingConfig",
    "E2FormerConfig",
    # Utilities
    "construct_radius_neighbor",
    "no_weight_decay",
    "create_trans_block",
    "get_powers",
    "irreps_times",
    "init_embeddings",
]