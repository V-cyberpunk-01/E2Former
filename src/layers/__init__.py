"""
E2Former Layers Module

Contains neural network layers and modules.
"""

# Import non-circular dependencies only
from .embeddings import (
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_higherorder_v3,
    EdgeDegreeEmbeddingNetwork_eqv2,
    BOOEmbedding,
    CoefficientMapping,
)
from .blocks import (
    TransBlock,
    MessageBlock_escn,
    MessageBlock_eqv2,
    construct_radius_neighbor,
)
from .interaction_blocks import (
    Body2_interaction,
    Body3_interaction_MACE,
)
from .dit import E2DiTBlock, TransformerBlock

# Note: attention module imports are handled directly where needed to avoid circular dependencies

__all__ = [
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
    # Others
    "E2DiTBlock",
    "TransformerBlock",
    "construct_radius_neighbor",
]