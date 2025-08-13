# -*- coding: utf-8 -*-
"""
E2Former Main Model Implementation

This file contains the main E2former class for molecular property prediction.

Author: E2Former Team
License: MIT
"""

import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from torch import logical_not, nn

# Import from modular files
from ..core.base_modules import (
    create_trans_block,
    init_embeddings,
    no_weight_decay,
)
from ..layers.attention import E2TensorProductArbitraryOrder
from ..layers.embeddings import (
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_eqv2,
)
from ..layers.blocks import construct_radius_neighbor
from ..core.module_utils import (
    GaussianRadialBasisLayer,
    GaussianSmearing,
    get_normalization_layer,
)

# FairChem imports
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

# Constants
DEFAULT_ATOM_TYPE_COUNT = 256


def get_powers(vec, coeffs, lmax):
    """Compute spherical harmonics powers for given vector.
    
    Args:
        vec: Input vector tensor
        coeffs: Coefficients from E2TensorProductArbitraryOrder.get_coeffs()
        lmax: Maximum l order
        
    Returns:
        List of computed powers for each l order
    """
    out_powers = [
        coeffs[0] * torch.ones_like(vec.narrow(-1, 0, 1).unsqueeze(dim=-1))
    ]
    # Y is pos. Precompute spherical harmonics for all orders
    for i in range(1, lmax + 1):
        out_powers.append(
            coeffs[i]
            * e3nn.o3.spherical_harmonics(
                i, vec, normalize=False, normalization="integral"
            ).unsqueeze(-1)
        )

    return out_powers


class E2former(torch.nn.Module):
    """E2Former: Equivariant Transformer for molecular property prediction.
    
    An E(3)-equivariant graph neural network that combines attention mechanisms
    with spherical harmonics and tensor products to predict molecular properties
    while maintaining rotational and translational equivariance.
    
    The model processes atomic graphs with positions and types to predict
    energy and forces (when enabled) using multiple transformer-like blocks
    with equivariant attention mechanisms.
    
    Methods are organized into:
    - Initialization methods
    - Forward pass helper methods  
    - Main forward method
    """
    
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=6,
        pbc_max_radius=15,
        max_neighbors=20,
        max_radius=15.0,
        basis_type="gaussiansmear",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+32x1e+32x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="rms_norm_sh",  # the default is deprecated
        alpha_drop=0.1,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.1,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        tp_type="QK_alpha",
        attn_type="first-order",
        edge_embedtype="default",
        attn_biastype="share",  # add
        ffn_type="default",
        add_rope=True,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        avg_degree=23.01,
        force_head=None,
        decouple_EF=False,
        **kwargs,
    ):
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.pbc_max_radius = pbc_max_radius  #
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.add_rope = add_rope
        self.time_embed = time_embed
        self.sparse_attn = sparse_attn
        self.dynamic_sparse_attn_threthod = dynamic_sparse_attn_threthod
        
        if pbc_max_radius != max_radius:
            raise ValueError("Please ensure these two radius equal for pbc and non-pbc generalize")
            
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.decouple_EF = decouple_EF
        
        if "0e" not in self.irreps_node_embedding:
            raise ValueError("Sorry, the irreps node embedding must have 0e embedding")

        self.unifiedtokentoembedding = nn.Linear(
            self.irreps_node_embedding[0][0], self.irreps_node_embedding[0][0]
        )

        self.default_node_embedding = torch.nn.Embedding(
            atom_type_cnt, self.irreps_node_embedding[0][0]
        )

        self._node_scalar_dim = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        ## this is for f( r_ij )
        self.basis_type = basis_type
        self.attn_biastype = attn_biastype
        self.heads2basis = nn.Linear(
            self.num_attn_heads, self.number_of_basis, bias=True
        )
        
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        elif self.basis_type == "gaussiansmear":
            self.rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        else:
            raise ValueError(f"Invalid basis_type: '{self.basis_type}'. Expected 'gaussiansmear' or 'bessel'")

        # edge embedding network
        if (
            "default" in edge_embedtype
            or "highorder" in edge_embedtype
            or "elec" in edge_embedtype
        ):
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                time_embed=self.time_embed,
                use_atom_edge=True,
                use_layer_norm="wolayernorm" not in edge_embedtype,
            )
        elif "eqv2" in edge_embedtype:
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                lmax=len(self.irreps_node_embedding) - 1,
                time_embed=self.time_embed,
            )
        else:
            raise ValueError(f"Invalid edge_embedtype: '{edge_embedtype}'. Please check edge embedtype")

        # Create transformer blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            blk = create_trans_block(
                irreps_node_embedding=self.irreps_node_embedding,
                number_of_basis=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
                layer_id=i,
                is_last_layer=(i == self.num_layers - 1),
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
            )
            self.blocks.append(blk)

        self.energy_force_block = None
        if self.decouple_EF:
            self.energy_force_block = create_trans_block(
                irreps_node_embedding=self.irreps_node_embedding,
                number_of_basis=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
                layer_id=0,
                is_last_layer=True,
                force_attn_type="first-order",
            )

        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.lmax = len(self.irreps_node_embedding) - 1
        
        self.norm_tmp = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.norm_final = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        
        if len(self.irreps_node_embedding) == 1:
            self.f_linear = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim),
                nn.LayerNorm(self.scalar_dim),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, 3 * self.scalar_dim),
            )

        self.apply(self._init_weights)

    def reset_parameters(self):
        warnings.warn("Sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        mixed_attn_bias=None,
        padding_mask: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[torch.Tensor] = None,
        return_node_irreps=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the E2former class.
        
        Args:
            batched_data (Dict): Input data containing positions, atomic numbers, etc.
            token_embedding (torch.Tensor): Input token embeddings, [B, L, D]
            mixed_attn_bias: Optional attention bias
            padding_mask (torch.Tensor): Padding mask, [B, L]
            pbc_expand_batched (Optional[Dict]): PBC expansion data
            time_embed (Optional[torch.Tensor]): Time embeddings
            return_node_irreps (bool): Whether to return node irreps
            
        Returns:
            torch.Tensor: Output tensor with node attributes and vectors
        """

        tensortype = self.default_node_embedding.weight.dtype
        device = batched_data["pos"].device
        B, L = batched_data["pos"].shape[:2]

        node_pos = batched_data["pos"]
        padding_mask = ~batched_data["atom_masks"]
 
        # node_pos.requires_grad = True
        node_pos = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, node_pos
        )

        if (time_embed is not None) and self.time_embed:
            time_embed = time_embed.to(dtype=tensortype)
        else:
            time_embed = None

        node_mask = logical_not(padding_mask)
        atomic_numbers = batched_data["atomic_numbers"].reshape(B, L)[node_mask]
        ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
        f_node_pos = node_pos[node_mask]
        f_N1 = f_node_pos.shape[0]
        f_batch = torch.arange(B).reshape(B, 1).repeat(1, L).to(device)[node_mask]

        expand_node_mask = node_mask
        expand_node_pos = node_pos
        outcell_index = torch.arange(L).unsqueeze(dim=0).repeat(B, 1).to(device)
        f_exp_node_pos = f_node_pos
        f_outcell_index = torch.arange(len(f_node_pos)).to(device)
        mol_type = 0  # torch.any(batched_data["is_molecule"]):
        L2 = L
        
        if torch.any(batched_data["pbc"]):
            mol_type = 1
            L2 = pbc_expand_batched["outcell_index"].shape[1]
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_node_pos = pbc_expand_batched["expand_pos"].float()
            expand_node_pos[
                pbc_expand_batched["expand_mask"]
            ] = 999  # set expand node pos padding to 9999
            expand_node_mask = logical_not(pbc_expand_batched["expand_mask"])

            f_exp_node_pos = expand_node_pos[expand_node_mask]
            f_outcell_index = (outcell_index + ptr[:B, None])[
                expand_node_mask
            ]  # e.g. n1*hidden [flatten_outcell_index] -> n2*hidden

        batched_data["mol_type"] = mol_type

        neighbor_info = construct_radius_neighbor(
            node_pos, node_mask,
            expand_node_pos, expand_node_mask,
            radius=self.max_radius,
            outcell_index=outcell_index,
            max_neighbors=self.max_neighbors
        )
        batched_data.update(neighbor_info)
        
        f_edge_vec = neighbor_info["f_edge_vec"]
        f_dist = neighbor_info["f_dist"]
        f_poly_dist = neighbor_info["f_poly_dist"]
        f_attn_mask = neighbor_info["f_attn_mask"]
        f_dist_embedding = self.rbf(f_dist)  # flattn_N* topK* self.number_of_basis)

        # node_mask is used for node_embedding -> f_N*hidden
        if token_embedding is not None:
            f_atom_embedding = self.unifiedtokentoembedding(
                token_embedding[node_mask]
            )  # [L, B, D] => [B, L, D]
        else:
            f_atom_embedding = self.default_node_embedding(atomic_numbers)

        coeffs = E2TensorProductArbitraryOrder.get_coeffs()
        
        batched_data.update(
            {
                "f_exp_node_pos": f_exp_node_pos,
                "f_outcell_index": f_outcell_index,
                "Y_powers": get_powers(f_node_pos, coeffs, self.lmax),
                "exp_Y_powers": get_powers(f_exp_node_pos, coeffs, self.lmax),
                "edge_vec_powers": torch.cat(get_powers(f_edge_vec, coeffs, self.lmax), dim=-2),
            }
        )
        
        # Edge degree embedding
        edge_degree_embedding_dense = self.edge_deg_embed_dense(
            f_atom_embedding,
            f_node_pos,
            f_dist,
            edge_scalars=f_dist_embedding,
            edge_vec=f_edge_vec,
            batch=None,
            attn_mask=f_attn_mask,
            atomic_numbers=atomic_numbers,
            batched_data=batched_data,
            time_embed=time_embed,
        )

        f_node_irreps = edge_degree_embedding_dense
        f_node_irreps[:, 0, :] = f_node_irreps[:, 0, :] + f_atom_embedding
        node_irreps_his = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )

        # Forward through transformer blocks
        for i, blk in enumerate(self.blocks):
            f_node_irreps, f_dist_embedding = blk(
                node_pos=f_node_pos,
                node_irreps=f_node_irreps,
                edge_dis=f_dist,
                poly_dist=f_poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=f_attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                batch=f_batch,
            )
            if i == len(self.blocks) - 2:
                node_irreps_his[node_mask] = self.norm_tmp(
                    f_node_irreps
                )  # the part of order 0

        f_node_irreps_final = self.norm_final(f_node_irreps)
        node_irreps = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps  # the part of order 0

        node_attr = torch.zeros((B, L, self._node_scalar_dim), device=device)
        node_vec = torch.zeros((B, L, 3, self._node_scalar_dim), device=device)
        
        if not self.decouple_EF:
            node_attr[node_mask] = f_node_irreps_final[:, 0]
            if f_node_irreps_final.shape[1] > 1:
                node_vec[node_mask] = f_node_irreps_final[:, 1:4]  # the part of order 0
        else:
            node_attr[node_mask] = self.energy_force_block.ffn_s2(f_node_irreps_final)[
                :, 0
            ]
            node_vec[node_mask] = self.energy_force_block.ga(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps_final,
                edge_dis=f_dist,
                poly_dist=f_poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=f_attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
            )[0][:, 1:4]
            
        if return_node_irreps:
            return node_attr, node_vec, node_irreps, node_irreps_his

        return node_attr, node_vec