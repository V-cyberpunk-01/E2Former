# -*- coding: utf-8 -*-
"""
E2Former Cluster Implementation

This file contains the E2formerCluster class and its specialized attention module
for handling cluster-based attention mechanisms with E(3) equivariance.

Extracted from e2former_modular.py for better modularity.

Author: E2Former Team
License: MIT
"""

import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from sklearn.cluster import KMeans
from torch import logical_not, nn
from torch_scatter import scatter_mean

# Import from newly created modular files
from ..core.base_modules import (
    create_trans_block,
    init_embeddings, 
    irreps_times,
    no_weight_decay,
)
from ..layers.attention import E2TensorProductArbitraryOrder
from ..layers.embeddings import EdgeDegreeEmbeddingNetwork_higherorder, EdgeDegreeEmbeddingNetwork_eqv2
from ..layers.blocks import (
    TransBlock,
    construct_radius_neighbor,
)
from ..core.module_utils import (
    FeedForwardNetwork_s3,
    DropPath_BL,
    RadialFunction,
    SO3_Linear_e2former,
    SO3_Linear2Scalar_e2former,
    SmoothLeakyReLU,
    Learn_PolynomialDistance,
    GaussianRadialBasisLayer,
    GaussianSmearing,
    get_normalization_layer,
    polynomial,
)

# FairChem imports
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

# Constants
DEFAULT_ATOM_TYPE_COUNT = 256
DEFAULT_HIDDEN_DIM = 128


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


class E2AttentionArbOrder_sparse_forcluster(torch.nn.Module):
    """
    Sparse attention module specifically designed for cluster operations.
    
    This attention mechanism is used in E2formerCluster to handle interactions
    between atomic nodes and cluster centroids with E(3) equivariance.
    """
    
    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order", ## second-order
        add_rope=True,
        layer_id = 0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        norm_layer = 'identity',
        with_cluster=False,
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.with_cluster = with_cluster
        irreps_head = e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        
        self.irreps_head = irreps_head
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        self.lmax = self.irreps_node_input[-1][1][0]
        # new params
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_smooth_softmax = "use_smooth_softmax" in tp_type 

        self.node_embed_dim = DEFAULT_HIDDEN_DIM

        self.source_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        self.target_embedding = nn.Embedding(
                self.atom_type_cnt, self.node_embed_dim
            )
        init_embeddings(self.source_embedding, self.target_embedding)

        self.alpha_act = SmoothLeakyReLU(0.2)
        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [attn_weight_input_dim + self.node_embed_dim + self.scalar_dim,
                                  min(128,attn_weight_input_dim//2),
                                  min(128,attn_weight_input_dim//2)]
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)


        if self.tp_type.startswith("dot_alpha"):
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                attn_weight_input_dim,
                lmax=self.lmax,
            )
            self.alpha_norm = torch.nn.LayerNorm(self.attn_scalar_head)
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)


            self.fc_m0 = nn.Linear(2*self.attn_weight_input_dim*(self.lmax+1),self.num_attn_heads * self.attn_scalar_head)
            self.rad_func_m0 = RadialFunction(self.edge_channel_list+
                                        [2 * self.attn_weight_input_dim * (self.lmax + 1)])
        elif self.tp_type.startswith("dot_alpha_small"):
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                attn_weight_input_dim//8,
                lmax=self.lmax,
            )
            self.alpha_norm = torch.nn.LayerNorm(self.attn_scalar_head)
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)


            self.fc_m0 = nn.Linear(2*self.attn_weight_input_dim//8*(self.lmax+1),self.num_attn_heads * self.attn_scalar_head)
            self.rad_func_m0 = RadialFunction(self.edge_channel_list+
                                        [2 * self.attn_weight_input_dim//8* (self.lmax + 1)])

        elif self.tp_type == "QK_alpha":
            self.query_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.key_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

            self.query_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.key_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            
            self.fc_easy = RadialFunction(self.edge_channel_list+[self.num_attn_heads])
        

        else:
            raise ValueError(f"Invalid tp_type: '{self.tp_type}'. Please check your tp_type")
        
        
        # self.gbf = GaussianLayer(self.attn_weight_input_dim)  # default output_dim = 128
        self.pos_embedding_proj = nn.Linear(self.attn_weight_input_dim, self.scalar_dim*2)
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim*2)

        self.poly1 = Learn_PolynomialDistance(degree = 1)
        self.poly2 = Learn_PolynomialDistance(degree = 2)

        if self.attn_type == "zero-order":
            self.rad_func_intputhead = RadialFunction(self.edge_channel_list+
                                                [self.irreps_node_input[0][0]])
        
            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
            
            
        elif self.attn_type == "first-order":
            self.rad_func_intputhead = RadialFunction(self.edge_channel_list+
                                            [self.irreps_node_input[0][0]])
        
            self.first_order_tp = E2TensorProductArbitraryOrder(self.irreps_node_input, 
                                                                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                                                                order = 1,
                                                                head = self.irreps_node_input[0][0],
                                                                learnable_weight=True,
                                                                connection_mode='uvw'
                                                            )
            
            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
        
        elif self.attn_type == "second-order":

            self.rad_func_intputhead = RadialFunction(self.edge_channel_list+
                                            [self.irreps_node_input[0][0]//2])
            self.proj_value = SO3_Linear_e2former(
                    self.irreps_node_input[0][0],
                    self.irreps_node_input[0][0]//2,
                    lmax=self.lmax,
                )
            # self.second_order_tp = E2TensorProductArbitraryOrder_woequal(self.irreps_node_input, 
            self.second_order_tp = E2TensorProductArbitraryOrder(irreps_times(self.irreps_node_input,0.5), 
                                                            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                                                            order = 2,
                                                            head = self.irreps_node_input[0][0]//2,
                                                            learnable_weight=True,
                                                            connection_mode='uvw'
                                                            )
            self.proj_sec = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
            

        
    @staticmethod
    def vector_rejection(vec, d_ij):
        r"""Computes the component of :obj:`vec` orthogonal to :obj:`d_ij`.

        Args:
            vec (torch.Tensor): The input vector.
            d_ij (torch.Tensor): The reference vector.
        """
        vec_proj = (vec * d_ij).sum(dim=-2, keepdim=True)
        return vec - vec_proj * d_ij


    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        cluster_pos=None,
        cluster_irreps_input=None,
        **kwargs,
    ):
        """
        Forward pass for cluster attention mechanism.
        
        Args:
            node_pos: Node positions
            node_irreps_input: Node irreducible representation features  
            edge_dis: Edge distances
            edge_vec: Edge vectors
            attn_weight: Attention weights from radial basis functions
            atomic_numbers: Atomic numbers for embeddings
            poly_dist: Polynomial distance features
            attn_mask: Attention mask
            batch: Batch indices
            batched_data: Dictionary containing additional batch data
            cluster_pos: Cluster positions
            cluster_irreps_input: Cluster irreducible representation features
            
        Returns:
            tuple: (updated_node_features, attention_weights)
        """
        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        f_N1, _, hidden = node_irreps_input.shape
        # f_N2 = 
        topK = attn_weight.shape[1]


        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        edge_feature = attn_weight.sum(dim=1)  # B*L*-1
        # print(node_irreps_input.shape,torch.ones_like(node_irreps_input[:,:,:1,:1]).shape,self.tp_weight(node_scalars).shape)
        # value = self.value_tp(node_irreps_input,torch.ones_like(node_irreps_input[:,:1,:1]),self.tp_weight(node_scalars))

        # src_node = self.source_embedding(atomic_numbers)
        tgt_node = self.target_embedding(atomic_numbers)
        

        # sparse_indices = batched_data["batchANDneighbor_indices"]
        # topK = sparse_indices[0].shape[2]
        x_edge = torch.cat(
                [
                    attn_weight,
                    tgt_node.reshape(f_N1, 1, -1).repeat(1,topK,1),
                    cluster_irreps_input[:,0][batched_data["f_sparse_idx_expnode"]],
                ],
                dim=-1,
            )
        
        
        key = cluster_irreps_input
        value = cluster_irreps_input
        x_0_extra = []
        if self.tp_type == 'dot_alpha' or self.tp_type == 'dot_alpha_small':
            node_irreps_input_dot = self.dot_linear(node_irreps_input)
            
            key = self.dot_linear(key)
            
            
            for l in range(self.lmax+1):
                rij_l = e3nn.o3.spherical_harmonics(l,edge_vec,normalize=True).unsqueeze(dim = -1) #B*N*N*2l+1*1
                
                node_l = node_irreps_input_dot[:,l**2:(l+1)**2] #B*N*2l+1*hidden
                slice_key = key[:,(l**2):((l+1)**2)] #B*N*2l+1*hidden
                # print(rij_l.shape,node_l.shape,node_irreps_input.shape)
                x_0_extra.append(torch.sum(rij_l*node_l.unsqueeze(dim=1),dim = -2))
                x_0_extra.append(torch.sum(rij_l*slice_key[batched_data["f_sparse_idx_expnode"]],dim = -2))
        
        
        if self.tp_type == 'QK_alpha':
            ## QK alpha
            query = self.query_linear(node_irreps_input).reshape(f_N1,self.num_attn_heads,-1)
            key = self.key_linear(key)


            key = key[batched_data["f_sparse_idx_expnode"]]
            key = key.reshape(f_N1,topK,self.num_attn_heads,-1)                 
            
            alpha = self.alpha_act(
                self.fc_easy(x_edge) * torch.sum(query.unsqueeze(dim=1)*key,dim = 3)/math.sqrt(query.shape[-1]))

        elif self.tp_type.startswith('dot_alpha') :
            edge_m0 = self.rad_func_m0(x_edge)

            x_0_alpha = self.fc_m0(torch.cat(x_0_extra,dim = -1)*edge_m0)
            x_0_alpha = x_0_alpha.reshape(f_N1,-1, self.num_attn_heads, self.attn_scalar_head)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("qeik, ik -> qei", x_0_alpha, self.alpha_dot)

        if self.use_smooth_softmax:
            alpha = alpha.to(torch.float64)
            poly_dist = poly_dist.to(alpha.dtype)
            alpha = alpha -  alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
            alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3)).to(torch.float32)
        else:
            alpha = alpha.masked_fill(attn_mask, -1e6)
            #######################biggest bug here!
            # alpha = torch.nn.functional.softmax(alpha, 2)
            alpha = torch.nn.functional.softmax(alpha, 1)
            alpha = alpha.masked_fill(attn_mask, 0)
            

        # alpha = alpha*x_0_extra_wosm
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        
        alpha_org = alpha
        
        edge_dis = edge_dis
        if self.attn_type != "all-order":
            inputhead  = self.rad_func_intputhead(x_edge)
            alpha = alpha.reshape(f_N1,-1,self.num_attn_heads,1) * inputhead.reshape(alpha.shape[:2]+
                                                                                    (self.num_attn_heads,-1)
                                                                                    )
            alpha = alpha.reshape(alpha.shape[:2]+(-1,))

            if self.attn_type == 'zero-order':
                node_output = self.proj_zero(torch.sum(alpha.unsqueeze(dim = 2)*value[batched_data["f_sparse_idx_expnode"]],dim = 1))
                
            if self.attn_type == 'first-order':
                node_output = self.proj_first(
                                    self.first_order_tp(node_pos,
                                        batched_data["f_cluster_pos"],
                                        None,
                                        value, 
                                        alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                                        batched_data["f_sparse_idx_expnode"],
                                        batched_data=batched_data))
                
            if self.attn_type == 'second-order':
                value = self.proj_value(value)
                node_output = self.proj_sec(
                                self.second_order_tp(node_pos,
                                        batched_data["f_cluster_pos"],
                                        None,
                                        value, 
                                        alpha / (edge_dis.unsqueeze(dim=-1)**2 + 1e-8),
                                        batched_data["f_sparse_idx_expnode"],
                                        batched_data=batched_data))


        # updated_attn_weight = attn_weight + node_irreps_input[:,:1].reshape(f_N1, 1, -1).repeat(1,topK,1) + node_irreps_input[:,0][f_sparse_idx_node]
        # updated_attn_weight = attn_weight + self.edge_updater(updated_attn_weight)
        return node_output,attn_weight


class E2formerCluster(torch.nn.Module):
    """
    E2Former with Clustering: Enhanced E2Former with hierarchical clustering capabilities.
    
    This model extends the basic E2Former with cluster-based attention mechanisms
    to handle long-range interactions more efficiently. It uses K-means clustering
    to create super-nodes and processes both atomic-level and cluster-level interactions.
    
    Key Features:
    - Hierarchical processing with cluster centroids
    - Long-short range interaction modeling  
    - Maintains E(3) equivariance at all levels
    - Adaptive clustering based on molecular size
    """
    
    def __init__(
        self,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        pbc_max_radius=5,
        max_neighbors=20,
        max_radius=15.0,
        basis_type="gaussian",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+16x1e+8x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="layer",  # the default is deprecated
        alpha_drop=0.1,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.1,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        tp_type=None,
        attn_type="v0",
        edge_embedtype="default",
        attn_biastype="share",  # add
        ffn_type="default",
        add_rope=True,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        avg_degree=23.01,  # _AVG_DEGREE
        force_head=None,
        decouple_EF=False,
        adptive_edge=False,
        long_short_layers=2,
        # mean=None,
        # std=None,
        # scale=None,
        # atomref=None,
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
        self.adptive_edge = adptive_edge
        # self.task_mean = mean
        # self.task_std = std
        # self.scale = scale
        # self.register_buffer("atomref", atomref)

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.decouple_EF = decouple_EF
        if "0e" not in self.irreps_node_embedding:
            raise ValueError("sorry, the irreps node embedding must have 0e embedding")

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
        # elif self.basis_type == "gaussian_edge":
        #     self.rbf = GaussianLayer_Edgetype(
        #         self.number_of_basis, cutoff=self.max_radius
        #     )
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

        # edge
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

        # self.norm_final = get_norm_layer(norm_layer)(
        #     o3.Irreps(f"{self.scalar_dim}x0e+{self.scalar_dim}x1e")
        # )
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

        ## cluster part 
        self.long_short_layers = long_short_layers
        self.irreps_node_input = o3.Irreps(irreps_node_embedding)
        self.irreps_node_output = o3.Irreps(irreps_node_embedding)
        
        # self.max_neighbors = max_neighbors
        # self.lmax = self.irreps_node_input[-1][1][0]
        # self.norm_1 = get_normalization_layer(
        #     norm_layer, lmax=self.lmax, num_channels=self.irreps_node_input[0][0]
        # )
        self.norm_cluster = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.rbf_cluster = GaussianSmearing(
                self.number_of_basis, cutoff=20, basis_width_scalar=2
            )
        # attn_weight_input_dim = self.number_of_basis

        self.cluster_blocks = torch.nn.ModuleList()
        self.norm_ffn = torch.nn.ModuleList()
        for i in range(self.long_short_layers):
            ga = E2AttentionArbOrder_sparse_forcluster(
                self.irreps_node_input,
                self.number_of_basis,
                num_attn_heads,
                attn_scalar_head,
                irreps_head,
                rescale_degree=rescale_degree,
                nonlinear_message=nonlinear_message,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                layer_id=0,
                attn_type=attn_type,
                tp_type=tp_type,
                add_rope=add_rope,
                sparse_attn=sparse_attn,
            )
            self.cluster_blocks.append(ga)
            self.norm_ffn.append(nn.Sequential(
                                    get_normalization_layer(
                                                norm_layer, lmax=self.lmax, num_channels=self.irreps_node_input[0][0]
                                                ),
                                    FeedForwardNetwork_s3(self.irreps_node_input[0][0],
                                                self.irreps_node_input[0][0],
                                                self.irreps_node_input[0][0],
                                                lmax = self.lmax,
                                                )))
        self.drop_path = None
        if drop_path_rate > 0.0:
            self.drop_path = DropPath_BL(drop_path_rate)

        self.apply(self._init_weights)

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            # # if self.weight_init == "normal":
            # std = 1 / math.sqrt(m.in_features)
            # torch.nn.init.normal_(m.weight, 0, std)
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
        Forward pass of the E2formerCluster class.
        
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
            
        The forward pass includes:
        1. Standard E2Former processing layers
        2. K-means clustering to create super-nodes
        3. Cluster-atom interaction processing
        4. Final output generation
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
                torch.zeros(1,dtype = torch.int32,device=device),
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
            #  batched_data["outcell_index"] # B*L2
            # batched_data["outcell_index_0"] # B*L2
            # batched_data.update(pbc_expand_batched)
            L2 = pbc_expand_batched["outcell_index"].shape[1]
            outcell_index = pbc_expand_batched["outcell_index"]
            # outcell_index_0 = (torch.arange(B).reshape(B, 1).repeat(1,batched_data["outcell_index"].shape[1] ))
            expand_node_pos = pbc_expand_batched["expand_pos"].float()
            expand_node_pos[
                pbc_expand_batched["expand_mask"]
            ] = 999  # set expand node pos padding to 9999
            expand_node_mask = logical_not(pbc_expand_batched["expand_mask"])

            f_exp_node_pos = expand_node_pos[expand_node_mask]
            f_outcell_index = (outcell_index + ptr[:B, None])[
                expand_node_mask
            ]  # e.g. n1*hidden [flatten_outcell_index]  -> n2*hidden

        batched_data["mol_type"] = mol_type

        neighbor_info = construct_radius_neighbor(node_pos,node_mask,
                              expand_node_pos,expand_node_mask,
                              radius = self.max_radius,
                              outcell_index = outcell_index,
                              max_neighbors = self.max_neighbors)
        batched_data.update(neighbor_info)
        
        f_edge_vec= neighbor_info["f_edge_vec"]
        f_dist= neighbor_info["f_dist"]
        f_poly_dist= neighbor_info["f_poly_dist"]
        f_attn_mask = neighbor_info["f_attn_mask"]
        f_dist_embedding = self.rbf(f_dist)  # flattn_N* topK* self.number_of_basis)

        # print(torch.max(f_sparse_idx_node),torch.max(f_sparse_idx_expnode),torch.max(ptr),torch.max(expand_ptr))
        # # this line could use to check the index's correctness
        # batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, L, topK)
        # test_edge_vec = (node_pos[:,:L].unsqueeze(dim = 2)-expand_node_pos[batch_indices,neighbor_indices])[node_mask]
        # print('test edge vec ',torch.sum(torch.abs(edge_vec-test_edge_vec)[~attn_mask.squeeze()]))

        # # this line could use to check the index's correctness
        # batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, L, topK)
        # test_src_ne = atomic_numbers[(torch.arange(B).reshape(B, 1).repeat(1,L2)),
        #                              outcell_index][batch_indices,neighbor_indices][node_mask]
        # src_ne = atomic_numbers[node_mask][flatten_sparse_indices_innode]
        # print('test atomic numbers',torch.sum(torch.abs(test_src_ne-src_ne)[~attn_mask.squeeze()]))

        # node_mask is used for node_embedding -> f_N*hidden
        # f_node_irreps = token_embedding[node_mask]
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
                # "edge_inter_mask": edge_inter_mask,  # used for escaip attention
                "Y_powers": get_powers(f_node_pos,coeffs,self.lmax),
                "exp_Y_powers": get_powers(f_exp_node_pos,coeffs,self.lmax),
                "edge_vec_powers": torch.cat(get_powers(f_edge_vec,coeffs,self.lmax),dim = -2),
            }
        )
        # f_N1 = f_node_pos.shape[0]
        # edge_mask = ~attn_mask.reshape(-1)
        # data = {"pos":f_node_pos,
        #         "atomic_numbers":atomic_numbers,
        #         "batch":f_batch,
        #         "natoms":torch.sum(node_mask,dim = -1),
        #         "node_offset":0,
        #         "atomic_numbers_full":atomic_numbers,
        #         "batch_full":f_batch,
        #         "edge_index":torch.stack([f_sparse_idx_node.reshape(-1).to(device)[edge_mask],
        #                       torch.arange(f_N1).reshape(f_N1,-1).repeat(1,topK).reshape(-1).to(device)[edge_mask]
        #                         ],dim = 0),
        #         "edge_distance":f_dist.reshape(f_N1*topK)[edge_mask],
        #         "edge_distance_vec":f_edge_vec.reshape(f_N1*topK,3)[edge_mask],

        #         }
        # return self.decoder(Data(**data))
        # if torch.any(torch.isnan(atom_embedding)):assert(False)
        # not use sparse mode
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
        # node_irreps = torch.zeros(B,L,9,self.irreps_node_embedding[0][0],device = device)
        f_node_irreps[:, 0, :] = f_node_irreps[:, 0, :] + f_atom_embedding
        node_irreps_his = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )

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
                batch=f_batch,  #
            )
            if i == len(self.blocks) - 2:
                node_irreps_his[node_mask] = self.norm_tmp(
                    f_node_irreps
                )  # the part of order 0

            # if torch.any(torch.isnan(node_irreps)):assert(False)
      
        f_node_irreps = self.norm_final(f_node_irreps)
        node_irreps = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps
        
        ########################## cluster part #################
        
        ### bulid cluster embeddings
        B, N, L, D = node_irreps.shape 
        node_irreps = node_irreps.reshape(B, N, -1) 
        cluster_ids = torch.zeros((B, N), dtype=torch.int, device=f_node_irreps.device)
        num_clusters_per_mol = torch.zeros((B), dtype=torch.int, device=f_node_irreps.device)

        for i in range(B):
            valid_pos = node_pos[i][node_mask[i]]  # [N_valid, 3]
            num_atoms = valid_pos.size(0)
            min_nodes_foreachGroup = min(15, num_atoms // 2 + 1)
            num_clusters = max(1, int(num_atoms / min_nodes_foreachGroup))
            num_clusters_per_mol[i] = num_clusters

            # KMeans 
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            cluster_ids_np = kmeans.fit_predict(valid_pos.cpu().detach().numpy())  # [N_valid]
            
            cluster_ids[i][node_mask[i]] = torch.tensor(cluster_ids_np, device=node_irreps.device)
        
        max_clusters = max(num_clusters_per_mol)
        expand_offset = torch.cumsum(
            torch.cat([torch.zeros(1, device=node_irreps.device, dtype=torch.long),
                    num_clusters_per_mol[:-1]]), dim=0
        ).unsqueeze(1)  # [B, 1] 

        global_cluster_ids = cluster_ids + expand_offset  # [B, N]
        flat_cluster_ids = global_cluster_ids[node_mask].reshape(-1)  # [B*N]
        
        # flat_embeddings =    # [B*N, D]
        # flat_pos = node_pos[node_mask].reshape(-1, 3)                  # [B*N, 3]
        f_cluster_pos = scatter_mean(f_node_pos, flat_cluster_ids, dim=0)                # [B*max_clusters, 3]
        
        # total_clusters = offset[-1] + num_clusters_per_mol[-1] # total global cluster count
        cluster_pos = torch.zeros(B, max_clusters, 3, device=node_irreps.device)
        cluster_mask = torch.zeros(B, max_clusters, dtype=torch.bool, device=node_irreps.device)

        # Mapping flat cluster index * batch and local cluster index
        batch_ids = torch.arange(B, device=node_irreps.device).repeat_interleave(num_clusters_per_mol)
        local_ids = torch.cat([torch.arange(n, device=node_irreps.device) for n in num_clusters_per_mol.tolist()])

        cluster_pos[batch_ids, local_ids] = f_cluster_pos
        cluster_mask[batch_ids, local_ids] = True

        # cluster_embeddings_padded = torch.zeros(B, max_clusters, D, device=node_irreps.device)
        # cluster_pos_padded = torch.zeros(B, max_clusters, 3, device=node_irreps.device)
        # cluster_mask = torch.zeros(B, max_clusters, dtype=torch.bool, device=node_irreps.device)

        # start = 0
        # for i in range(B):
        #     n_c = num_clusters_per_mol[i]
        #     cluster_embeddings_padded[i, :n_c] = cluster_embeddings[start:start + n_c]
        #     cluster_pos_padded[i, :n_c] = cluster_pos[start:start + n_c]
        #     cluster_mask[i, :n_c] = True
        #     start += n_c
        
        ### cluster - atom interaction
        node_irreps = node_irreps.reshape(B, N, L, D) 
        # node_irreps = self.norm_1(node_irreps)
        # device = padding_mask.device
        # B, L = padding_mask.shape[:2]

        # mol_type = batched_data["mol_type"]

        # B, L = padding_mask.shape[:2]
        # node_mask = logical_not(padding_mask)
        # f_node_pos = node_pos[node_mask]
        # f_batch = torch.arange(B).reshape(B,1).repeat(1,L).to(device)[node_mask]

        # L2 = cluster_pos.shape[1]
        # f_cluster_pos = cluster_pos[cluster_mask]
        # f_cluster_irreps = cluster_irreps[cluster_mask]
        expand_ptr = torch.cat([torch.Tensor([0,]).int().to(device),
                        torch.cumsum(torch.sum(cluster_mask,dim = -1),dim = -1)]
                        ,dim = 0)
        edge_vec = node_pos.unsqueeze(2) - cluster_pos.unsqueeze(1)
        dist = torch.norm(edge_vec, dim=-1)  # B*L*L Attention: ego-connection is 0 here
        mask = (dist < 4) | logical_not(cluster_mask.unsqueeze(1).expand(-1,N,-1)) | (padding_mask.unsqueeze(-1).expand(-1,-1,cluster_mask.shape[-1])) #(dist < 3.0) | 
        filtered_dist = torch.where(mask, 1e4, dist)

        # topK = (cluster_mask.shape[-1] + 2) // 2
        min_num_clusters = torch.min(num_clusters_per_mol)
        topK = min(min_num_clusters, self.max_neighbors)
        _, neighbor_indices = filtered_dist.sort(dim=-1, descending=False)
        neighbor_indices = neighbor_indices[:, :, :topK]

        dist = torch.gather(dist, dim=-1, index=neighbor_indices)  # Shape: B*L*topK
        attn_mask = (dist < (self.max_radius if mol_type != 1 else self.pbc_max_radius))
        attn_mask = attn_mask[node_mask].unsqueeze(dim = -1)

        f_dist = dist[node_mask] #flattn_N* topK*
        f_dist_embedding = self.rbf_cluster(f_dist) #flattn_N* topK* self.number_of_basis)
        poly_dist = polynomial(f_dist, self.max_radius if mol_type != 1 else self.pbc_max_radius)

        f_sparse_idx_expnode = (neighbor_indices+expand_ptr[:B,None,None])[node_mask]
        f_sparse_idx_expnode = torch.clamp(f_sparse_idx_expnode,max = expand_ptr[B]-1)
        f_edge_vec = f_node_pos.unsqueeze(dim = 1)-f_cluster_pos[f_sparse_idx_expnode]

        # coeffs = E2TensorProductArbitraryOrder.get_coeffs()
        # Y_powers = [coeffs[0]*torch.ones_like(f_node_pos.narrow(-1,0,1).unsqueeze(dim=-1))]
        # # Y is pos. Precompute spherical harmonics for all orders
        # for i in range(1,self.lmax + 1):
        #     Y_powers.append(coeffs[i]*e3nn.o3.spherical_harmonics(i, f_node_pos, normalize=False, normalization='integral').unsqueeze(-1))
        
        exp_Y_powers = [coeffs[0]*torch.ones_like(f_cluster_pos.narrow(-1,0,1).unsqueeze(dim=-1))]
        # Y is pos. Precompute spherical harmonics for all orders
        for i in range(1,self.lmax + 1):
            exp_Y_powers.append(coeffs[i]*e3nn.o3.spherical_harmonics(i, f_cluster_pos, normalize=False, normalization='integral').unsqueeze(-1))
        
        edge_vec_powers = [coeffs[0]*torch.ones_like(f_edge_vec.narrow(-1,0,1).unsqueeze(dim=-1))]
        # Y is pof_sparse_idx_expnodes. Precompute spherical harmonics for all orders
        for i in range(1,self.lmax + 1):
            edge_vec_powers.append(e3nn.o3.spherical_harmonics(i, f_edge_vec, normalize=True, normalization='integral').unsqueeze(-1))
        edge_vec_powers = torch.cat(edge_vec_powers,dim = -2)
        
        f_cluster_irreps = scatter_mean(f_node_irreps.reshape(-1, L*D), flat_cluster_ids, dim=0).reshape(-1, L, D)  # [B*max_clusters, D]       
        batched_data.update( 
            {
            'f_sparse_idx_expnode':f_sparse_idx_expnode,
            'f_cluster_pos':f_cluster_pos,
            'exp_h':f_cluster_irreps,
            # "Y_powers":Y_powers,
            "exp_Y_powers":exp_Y_powers,
            "edge_vec_powers":edge_vec_powers
            }
            )
        f_atomic_numbers = batched_data["atomic_numbers"].reshape(B, N)[node_mask] 
        
        for i, blk in enumerate(self.cluster_blocks):
            # f_cluster_irreps = scatter_mean(f_node_irreps.reshape(-1, L*D), flat_cluster_ids, dim=0).reshape(-1, L, D)  # [B*max_clusters, D]
            # batched_data.update({'exp_h':f_cluster_irreps,})
            f_node_irreps_res = f_node_irreps
            f_node_irreps, attn_weight = blk(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps,
                edge_dis=f_dist,
                poly_dist=poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=f_atomic_numbers,
                attn_mask=attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                cluster_irreps_input=f_cluster_irreps,
                batch = f_batch
            )
            if self.drop_path is not None:
                f_node_irreps = self.drop_path(f_node_irreps,f_batch)

            f_node_irreps = f_node_irreps + f_node_irreps_res
            f_node_irreps_res = f_node_irreps
            f_node_irreps = self.norm_ffn[i](f_node_irreps)
            if self.drop_path is not None:
                f_node_irreps = self.drop_path(f_node_irreps,f_batch)
            f_node_irreps = f_node_irreps_res + f_node_irreps

        f_node_irreps_final = self.norm_cluster(f_node_irreps)
        
        # output part
        
        node_irreps[node_mask] = f_node_irreps_final  # the part of order 0
        node_attr = torch.zeros((B, N, self._node_scalar_dim), device=device)
        node_vec = torch.zeros((B, N, 3, self._node_scalar_dim), device=device)
        if not self.decouple_EF:
            node_attr[node_mask] = f_node_irreps_final[:, 0]
            node_vec[node_mask] = f_node_irreps_final[:, 1:4]  # the part of order 0
        else:
            node_attr[node_mask] = self.energy_force_block.ffn_s2(f_node_irreps_final)[
                :, 0
            ]
            node_vec[node_mask] = self.energy_force_block.ga(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps_final,
                edge_dis=f_dist,
                poly_dist=poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
            )[0][:, 1:4]
        if return_node_irreps:
            return node_attr, node_vec, node_irreps, node_irreps_his

        return node_attr, node_vec