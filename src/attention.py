# -*- coding: utf-8 -*-
"""
E2Former Attention Mechanisms

This module contains the sparse attention mechanisms for E2Former, including:
- E2AttentionArbOrder_sparse: Main sparse attention implementation
- E2AttentionArbOrder_sparse_forcluster: Cluster-aware sparse attention
"""

import math
import torch
import e3nn
from e3nn import o3
from torch import nn

# Import required modules
from .module_utils import (
    RadialFunction,
    SmoothLeakyReLU,
    Learn_PolynomialDistance,
    SO3_Linear_e2former,
    SO3_Linear2Scalar_e2former,
    get_normalization_layer,
)
from .wigner6j.tensor_product import E2TensorProductArbitraryOrder

# Constants
DEFAULT_ATOM_TYPE_COUNT = 256
DEFAULT_HIDDEN_DIM = 128
EMBEDDING_INIT_RANGE = (-0.001, 0.001)


def init_embeddings(
    source_embedding: nn.Embedding,
    target_embedding: nn.Embedding,
    init_range: tuple[float, float] = EMBEDDING_INIT_RANGE
) -> None:
    """Initialize source and target embeddings with uniform distribution.
    
    Args:
        source_embedding: Source embedding layer
        target_embedding: Target embedding layer  
        init_range: Range for uniform initialization
    """
    nn.init.uniform_(source_embedding.weight.data, *init_range)
    nn.init.uniform_(target_embedding.weight.data, *init_range)


def irreps_times(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    """Multiply the multiplicities of irreps by a factor.
    
    Args:
        irreps: Input irreducible representations
        factor: Multiplication factor for multiplicities
        
    Returns:
        New irreps with scaled multiplicities
    """
    out = [(int(mul * factor), ir) for mul, ir in irreps if mul > 0]
    return e3nn.o3.Irreps(out)


class E2AttentionArbOrder_sparse(torch.nn.Module):
    """
    Use IrrepsLinear with external weights W(|r_i|)

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
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        norm_layer="identity",
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
        irreps_head = (
            e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )

        self.irreps_head = irreps_head
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        self.lmax = self.irreps_node_input[-1][1][0]
        # new params
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_smooth_softmax = "use_smooth_softmax" in tp_type

        self.node_embed_dim = DEFAULT_HIDDEN_DIM

        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        init_embeddings(self.source_embedding, self.target_embedding)

        self.alpha_act = SmoothLeakyReLU(0.2)
        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [
            attn_weight_input_dim + self.node_embed_dim * 2,
            min(DEFAULT_HIDDEN_DIM, attn_weight_input_dim // 2),
            min(DEFAULT_HIDDEN_DIM, attn_weight_input_dim // 2),
        ]
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

            self.fc_m0 = nn.Linear(
                2 * self.attn_weight_input_dim * (self.lmax + 1),
                self.num_attn_heads * self.attn_scalar_head,
            )
            self.rad_func_m0 = RadialFunction(
                self.edge_channel_list
                + [2 * self.attn_weight_input_dim * (self.lmax + 1)]
            )
        elif self.tp_type.startswith("dot_alpha_small"):
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                attn_weight_input_dim // 8,
                lmax=self.lmax,
            )
            self.alpha_norm = torch.nn.LayerNorm(self.attn_scalar_head)
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

            self.fc_m0 = nn.Linear(
                2 * self.attn_weight_input_dim // 8 * (self.lmax + 1),
                self.num_attn_heads * self.attn_scalar_head,
            )
            self.rad_func_m0 = RadialFunction(
                self.edge_channel_list
                + [2 * self.attn_weight_input_dim // 8 * (self.lmax + 1)]
            )

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

            self.fc_easy = RadialFunction(
                self.edge_channel_list + [self.num_attn_heads]
            )

        else:
            raise ValueError(f"Invalid basis_type: '{self.basis_type}'. Expected 'gaussiansmear' or 'bessel'")("please check your tp_type")

        # self.gbf = GaussianLayer(self.attn_weight_input_dim)  # default output_dim = 128
        self.pos_embedding_proj = nn.Linear(
            self.attn_weight_input_dim, self.scalar_dim * 2
        )
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim * 2)

        self.poly1 = Learn_PolynomialDistance(degree=1)
        self.poly2 = Learn_PolynomialDistance(degree=2)

        if self.attn_type == "zero-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )

            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "first-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )

            self.first_order_tp = E2TensorProductArbitraryOrder(
                self.irreps_node_input,
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=1,
                head=self.irreps_node_input[0][0],
                learnable_weight=True,
                connection_mode="uvw",
            )

            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "second-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 2]
            )
            self.proj_value = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 2,
                lmax=self.lmax,
            )
            # self.second_order_tp = E2TensorProductArbitraryOrder_woequal(self.irreps_node_input,
            self.second_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.5),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=2,
                head=self.irreps_node_input[0][0] // 2,
                learnable_weight=True,
                connection_mode="uvw",
            )
            self.proj_sec = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "all-order":
            self.rad_func_intputhead_zero = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )
            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

            self.rad_func_intputhead_fir = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 2]
            )
            self.proj_value_fir = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 2,
                lmax=self.lmax,
            )
            self.first_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.5),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=1,
                head=self.irreps_node_input[0][0] // 2,
                learnable_weight=True,
                connection_mode="uvw",
            )

            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

            self.rad_func_intputhead_sec = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 4]
            )
            self.proj_value_sec = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 4,
                lmax=self.lmax,
            )
            # self.second_order_tp = E2TensorProductArbitraryOrder_woequal(self.irreps_node_input,
            self.second_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.25),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=2,
                head=self.irreps_node_input[0][0] // 4,
                learnable_weight=True,
                connection_mode="uvw",
            )
            self.proj_sec = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
        self.norm_1 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )
        self.norm_2 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )

        # self.edge_updater = RadialFunction([attn_weight_input_dim,
        #                                     min(128,attn_weight_input_dim//2),
        #                                     attn_weight_input_dim])

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
        node_pos: torch.Tensor,
        node_irreps_input: torch.Tensor,
        edge_dis: torch.Tensor,
        edge_vec: torch.Tensor,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        from molfm.models.psm.equivariant.e2former import *

        irreps_in = o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        attn_scalar_head = 32
        func = E2AttentionSecondOrder(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+32x1e+32x2e",
            alpha_drop=0.1,
            tp_type='easy_alpha'
        )
        out = func(node_pos,
            torch.randn(B,L,9,256),
            node_dis,
            torch.randn(B,L,L,3),
            torch.randn(B,L,L,attn_scalar_head),
            atomic_numbers = torch.randint(0,19,(B,L)),
            attn_mask = torch.randn(B,L,L,1)>0)
        """
        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        f_N1, _, hidden = node_irreps_input.shape
        # f_N2 =
        topK = attn_weight.shape[1]

        f_sparse_idx_node = batched_data["f_sparse_idx_node"]

        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        edge_feature = attn_weight.sum(dim=1)  # B*L*-1
        # print(node_irreps_input.shape,torch.ones_like(node_irreps_input[:,:,:1,:1]).shape,self.tp_weight(node_scalars).shape)
        # value = self.value_tp(node_irreps_input,torch.ones_like(node_irreps_input[:,:1,:1]),self.tp_weight(node_scalars))
        value = node_irreps_input  # *node_scalars[:,None]

        src_node = self.source_embedding(atomic_numbers)
        tgt_node = self.target_embedding(atomic_numbers)

        # sparse_indices = batched_data["batchANDneighbor_indices"]
        # topK = sparse_indices[0].shape[2]
        x_edge = torch.cat(
            [
                attn_weight,
                tgt_node.reshape(f_N1, 1, -1).repeat(1, topK, 1),
                src_node[f_sparse_idx_node],
            ],
            dim=-1,
        )

        x_0_extra = []
        if self.tp_type == "dot_alpha" or self.tp_type == "dot_alpha_small":
            node_irreps_input_dot = self.dot_linear(node_irreps_input)
            for l in range(self.lmax + 1):
                rij_l = e3nn.o3.spherical_harmonics(
                    l, edge_vec, normalize=True
                ).unsqueeze(
                    dim=-1
                )  # B*N*N*2l+1*1

                node_l = node_irreps_input_dot[
                    :, l**2 : (l + 1) ** 2
                ]  # B*N*2l+1*hidden
                # print(rij_l.shape,node_l.shape,node_irreps_input.shape)
                x_0_extra.append(torch.sum(rij_l * node_l.unsqueeze(dim=1), dim=-2))
                x_0_extra.append(torch.sum(rij_l * node_l[f_sparse_idx_node], dim=-2))

        if self.tp_type == "QK_alpha":
            ## QK alpha
            query = self.query_linear(node_irreps_input).reshape(
                f_N1, self.num_attn_heads, -1
            )
            key = self.key_linear(node_irreps_input)

            key = key.reshape(f_N1, self.num_attn_heads, -1)

            key = key[f_sparse_idx_node]

            alpha = self.alpha_act(
                self.fc_easy(x_edge)
                * torch.sum(query.unsqueeze(dim=1) * key, dim=3)
                / math.sqrt(query.shape[-1])
            )

        elif self.tp_type.startswith("dot_alpha"):
            edge_m0 = self.rad_func_m0(x_edge)

            x_0_alpha = self.fc_m0(torch.cat(x_0_extra, dim=-1) * edge_m0)
            x_0_alpha = x_0_alpha.reshape(
                f_N1, -1, self.num_attn_heads, self.attn_scalar_head
            )
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("qeik, ik -> qei", x_0_alpha, self.alpha_dot)

        # key = key[sparse_indices[0],sparse_indices[1]]
        # alpha = self.alpha_act(
        #     self.fc_easy(x_edge) * torch.einsum("bihd,bijhd->bijh",query,key)/math.sqrt(query.shape[-1]))

        if self.use_smooth_softmax:
            alpha = alpha.to(torch.float64)
            poly_dist = poly_dist.to(alpha.dtype)
            alpha = alpha - alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
            alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3)).to(
                torch.float32
            )
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
            inputhead = self.rad_func_intputhead(x_edge)
            alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
                alpha.shape[:2] + (self.num_attn_heads, -1)
            )
            alpha = alpha.reshape(alpha.shape[:2] + (-1,))
            # batched_data.update(
            #     {'f_sparse_idx_node':f_sparse_idx_node,
            #     'f_sparse_idx_expnode':f_sparse_idx_expnode,
            #     'f_exp_node_pos':f_exp_node_pos,
            #     'f_outcell_index':f_outcell_index
            #     }
            if self.attn_type == "zero-order":
                node_output = self.proj_zero(
                    torch.sum(
                        alpha.unsqueeze(dim=2)
                        * value[batched_data["f_sparse_idx_node"]],
                        dim=1,
                    )
                )

            if self.attn_type == "first-order":
                node_output = self.proj_first(
                    self.first_order_tp(
                        node_pos,
                        batched_data["f_exp_node_pos"],
                        None,
                        value[batched_data["f_outcell_index"]],
                        alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                        batched_data["f_sparse_idx_expnode"],
                        batched_data=batched_data,
                    )
                )

            if self.attn_type == "second-order":
                value = self.proj_value(value)
                node_output = self.proj_sec(
                    self.second_order_tp(
                        node_pos,
                        batched_data["f_exp_node_pos"],
                        None,
                        value[batched_data["f_outcell_index"]],
                        alpha / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                        batched_data["f_sparse_idx_expnode"],
                        batched_data=batched_data,
                    )
                )

        if self.attn_type == "all-order":
            node_gate = torch.nn.functional.sigmoid(
                self.pos_embedding_proj(edge_feature)
                + self.node_scalar_proj(node_irreps_input[:, 0, :])
            )

            inputhead_zero = self.rad_func_intputhead_zero(x_edge)
            alpha_zero = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_zero.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_zero = alpha_zero.reshape(alpha_zero.shape[:2] + (-1,))
            node_output_zero = self.proj_zero(
                torch.sum(
                    alpha_zero.unsqueeze(dim=2)
                    * value[batched_data["f_sparse_idx_node"]],
                    dim=1,
                )
            )

            # value = self.norm_1(value+node_output_zero)
            inputhead_fir = self.rad_func_intputhead_fir(x_edge)
            alpha_fir = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_fir.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_fir = alpha_fir.reshape(alpha_fir.shape[:2] + (-1,))
            node_output_fir = self.proj_first(
                self.first_order_tp(
                    node_pos,
                    batched_data["f_exp_node_pos"],
                    None,
                    self.proj_value_fir(value)[batched_data["f_outcell_index"]],
                    alpha_fir / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                    batched_data["f_sparse_idx_expnode"],
                    batched_data=batched_data,
                )
            )

            # value = self.norm_2(value+node_output_fir)

            inputhead_sec = self.rad_func_intputhead_sec(x_edge)
            alpha_sec = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_sec.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_sec = alpha_sec.reshape(alpha_sec.shape[:2] + (-1,))
            node_output_sec = self.proj_sec(
                self.second_order_tp(
                    node_pos,
                    batched_data["f_exp_node_pos"],
                    None,
                    self.proj_value_sec(value)[batched_data["f_outcell_index"]],
                    alpha_sec / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                    batched_data["f_sparse_idx_expnode"],
                    batched_data=batched_data,
                )
            )
            node_output = (
                node_output_zero * node_gate[:, None, : self.scalar_dim]
                + node_output_fir * node_gate[:, None, self.scalar_dim :]
                + node_output_sec * (1 - node_gate[:, None, self.scalar_dim :])
            )
            # node_output = node_output_own*(1-node_gate[:,None,:self.scalar_dim])+
        # updated_attn_weight = attn_weight + node_irreps_input[:,:1].reshape(f_N1, 1, -1).repeat(1,topK,1) + node_irreps_input[:,0][f_sparse_idx_node]
        # updated_attn_weight = attn_weight + self.edge_updater(updated_attn_weight)
        return node_output, attn_weight


class E2AttentionArbOrder_sparse_forcluster(torch.nn.Module):
    """
    Use IrrepsLinear with external weights W(|r_i|)

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
            raise ValueError(f"Invalid basis_type: '{self.basis_type}'. Expected 'gaussiansmear' or 'bessel'")("please check your tp_type")
        
        
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
        from molfm.models.psm.equivariant.e2former import *

        irreps_in = o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        attn_scalar_head = 32
        func = E2AttentionSecondOrder(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+32x1e+32x2e",
            alpha_drop=0.1,
            tp_type='easy_alpha'
        )
        out = func(node_pos,
            torch.randn(B,L,9,256),
            node_dis,
            torch.randn(B,L,L,3),
            torch.randn(B,L,L,attn_scalar_head),
            atomic_numbers = torch.randint(0,19,(B,L)),
            attn_mask = torch.randn(B,L,L,1)>0)
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