# -*- coding: utf-8 -*-
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3

from torch_geometric.data import Data


from .dit import E2DiTBlock,TransformerBlock
from .e2former import E2former, no_weight_decay, construct_radius_neighbor,E2formerCluster
from .E2Former_configs import E2FormerConfigs
from .module_utils import CellExpander,GaussianLayer_Edgetype,polynomial


from .utils.graph_utils import compilable_scatter, unpad_results,RandomRotate
from .utils.nn_utils import init_linear_weights
from .utils.base_utils import registry,init_configs


_AVG_NUM_NODES = 77.81317


def process_batch_data(data, max_nodes=None):
    """
    Process raw batch data into padded batched format with masks.

    Args:
        data: Input data containing pos, cell, atomic_numbers, etc.
        max_nodes: Maximum number of nodes for padding. If None, uses maximum in batch.

    Returns:
        dict: Contains batched and padded data with masks
    """
    if len(data.pos.shape) == 3:
        return data
    # otherwise, flatten_N * 3 format

    # Extract batch information
    batch_idx = data.batch
    num_graphs = data.ptr.size(0) - 1

    if max_nodes is None:
        max_nodes = max([data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)])

    # Initialize output tensors
    batched_pos = torch.zeros((num_graphs, max_nodes, 3), device=data.pos.device)
    batched_token_id = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    masked_token_type = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    padding_mask = torch.ones(
        (num_graphs, max_nodes), dtype=torch.bool, device=data.pos.device
    )
    # if "is_molecule" in data:

    # else:
    if "pbc" not in data:
        # pbc = torch.tensor([[1, 1, 1]]).repeat(num_graphs, 1).to(data.pos.device) # default for open catylst
        pbc = (
            torch.tensor([[1, 1, 0]]).repeat(num_graphs, 1).to(data.pos.device)
        )  # default for open catylst
    else:
        pbc = data.pbc

    is_protein = torch.zeros(
        (num_graphs, max_nodes, 1), dtype=torch.bool, device=data.pos.device
    )
    num_atoms = torch.tensor(
        [data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)],
        dtype=torch.long,
        device=data.pos.device,
    )

    # Process each graph in the batch
    for i in range(num_graphs):
        start_idx = data.ptr[i]
        end_idx = data.ptr[i + 1]
        num_nodes = end_idx - start_idx

        # Fill in positions
        batched_pos[i, :num_nodes] = data.pos[start_idx:end_idx]

        # Fill in cell (assuming one cell per graph)

        # Fill in atomic numbers (token_ids)
        batched_token_id[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]
        if "masked_token_type" in data:
            masked_token_type[i, :num_nodes] = data.masked_token_type[start_idx:end_idx]
        else:
            masked_token_type[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]

        # Set mask (False indicates valid entries)
        padding_mask[i, :num_nodes] = False

    batched_data = {
        "pos": batched_pos,  # [num_graphs, max_nodes, 3]
        "cell": data.cell,  # [num_graphs, 3, 3]
        "token_id": batched_token_id,  # [num_graphs, max_nodes]
        "masked_token_type": masked_token_type,
        "padding_mask": padding_mask,  # [num_graphs, max_nodes]
        "pbc": pbc,  # [num_graphs, 3]
        "subset_name": None if "subset_name" not in data else data.subset_name,
        "forces_subset_name": None
        if "forces_subset_name" not in data
        else data.forces_subset_name,  # "is_stable_periodic": is_stable_periodic,  # [num_graphs, 1]
        # "is_molecule": is_molecule,
        # "is_periodic": is_periodic,
        "is_protein": is_protein,
        "position_ids": torch.arange(max_nodes)
        .unsqueeze(dim=0)
        .repeat(
            num_graphs, 1
        ),  # unused parameter: only for protein, id in sequence for pos cos and sin embed
        "num_atoms": num_atoms,  # [num_graphs]
        "node_batch": batch_idx,  # [num_nodes]
        "graph_padding_mask": padding_mask,  # [num_graphs, max_nodes]
    }

    batched_data = Data(**batched_data)

    return batched_data


@registry.register_model("PSM_ESCAIP_backbone")
class E2FormerBackbone(nn.Module):
    """
    Physics Science Module backbone model integrated with EScAIP framework.

    This model combines the PSM architecture with EScAIP's configuration and processing
    pipeline, enabling it to work within the EScAIP framework while maintaining PSM's
    unique architectural features.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        # Load configs
        cfg = init_configs(E2FormerConfigs, kwargs)
        self.cfg = cfg
        self.global_cfg = cfg.global_cfg
        # self.molecular_graph_cfg = cfg.molecular_graph_cfg
        # self.gnn_cfg = cfg.gnn_cfg
        # self.reg_cfg = cfg.reg_cfg

        # Training configuration
        self.regress_forces = cfg.global_cfg.regress_forces

        # PSM specific configuration
        # TODO: Integrate PSM config with EScAIP config system
        self.psm_config = cfg.psm_config

        # Cell expansion for periodic boundary conditions
        self.cell_expander = CellExpander(
            self.kwargs.get("pbc_max_radius",5.),
            self.kwargs.get("expanded_token_cutoff", 512),  # deprecated
            self.kwargs.get("pbc_expanded_num_cell_per_direction",4),
            self.kwargs.get("pbc_max_radius",5.),
        )

        # Token embedding layer
        self.embedding = nn.Embedding(256, cfg.encoder_embed_dim)
        self.embedding_charge = nn.Embedding(30, cfg.encoder_embed_dim)
        self.embedding_multiplicity = nn.Embedding(30, cfg.encoder_embed_dim)

        # self.boo_embedding = nn.Embedding(256 + 300, cfg.encoder_embed_dim)
        # self.boo_embedding_linear = nn.Linear(cfg.encoder_embed_dim, 128)

        # Configuration loaded successfully

        self.uniform_center_count = 5
        self.sph_grid_channel = 8
        # self.linear_sigmaco = torch.nn.Sequential(
        #     nn.Linear(128 + 128 + 128, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 2 * self.uniform_center_count * self.sph_grid_channel),
        # )
        # self.electron_density = Electron_Density_Descriptor(
        #     uniform_center_count=self.uniform_center_count,
        #     num_sphere_points=16,
        #     channel=self.sph_grid_channel,
        #     lmax=2,
        #     output_channel=cfg.encoder_embed_dim,
        # )

        self.invariant_encoder = None
        if cfg.encoder == "dit":

            # DIT encoder layers initialized with config
            self.invariant_encoder = E2DiTBlock(
                    embedding_dim=cfg.encoder_embed_dim, 
                    radius_cut_off = self.kwargs.get("pbc_max_radius",5.),
                    max_neighbors = self.kwargs.get("max_neighbors",32),
                    num_layers = cfg.dit_config.num_encoder_layers,
                    **vars(cfg.dit_config)
                )
        elif cfg.encoder == "transformer":
            # Transformer encoder layers initialized with config
            self.invariant_encoder = TransformerBlock(
                    embedding_dim=cfg.encoder_embed_dim, 
                    radius_cut_off = self.kwargs.get("pbc_max_radius",5.),
                    max_neighbors = self.kwargs.get("max_neighbors",32),
                    num_layers = cfg.dit_config.num_encoder_layers,
                    **vars(cfg.dit_config))


        self.embed_proj = torch.nn.Sequential(
            nn.Linear(
                cfg.encoder_embed_dim,
                o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0],
            ),
            nn.SiLU(),
            nn.LayerNorm(
                o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0], eps=1e-6
            ),
        )

        self.fea_dim = o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0]
        # Decoder selection and initialization
        print("e2former use config like follows: \n", cfg.backbone_config)
        
        if cfg.backbone_config.with_cluster:
            self.decoder = E2formerCluster(**vars(cfg.backbone_config))
        else:
            self.decoder = E2former(**vars(cfg.backbone_config))
        # Enable high precision matrix multiplication if not using fp16
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")

        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)
        # print("compiled:", self.global_cfg.use_compile)

        # Set up forward function with optional compilation
        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def BOO_feature(self, pos, expand_pos, local_attention_weight):
        B, N1 = pos.shape[:2]
        expand_pos.shape[1]
        dist = torch.norm(pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1), dim=-1)
        edge_vec = (pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1)) / (
            dist.unsqueeze(dim=-1) + 1e-5
        )
        angel = torch.sum(
            edge_vec * (local_attention_weight.unsqueeze(dim=-1) > 1e-6), dim=2
        )
        # print('before norm',angel[0],torch.sum(local_attention_weight>1e-6,dim =2)[0])
        # angel = 2*angel/(torch.sum(local_attention_weight>1e-6,dim =2).unsqueeze(dim = -1))
        # angel = angel
        angel = torch.sum(angel**2, dim=-1) - torch.sum(
            local_attention_weight > 1e-6, dim=2
        ).unsqueeze(dim=-1)
        # angel = self.boo_embedding(torch.sum(local_attention_weight>1e-6,dim =2))
        # print('after norm',angel[0],torch.max(angel),torch.min(angel))
        return angel

    def compiled_forward(
        self,
        batched_data,
        **kwargs,
    ):
        """
        Forward pass implementation that can be compiled with torch.compile.
        """
        # Enable gradient computation for forces if needed
        use_grad = (
            True  # self.global_cfg.regress_forces and not self.global_cfg.direct_force
        )
        batched_data["pos"].requires_grad_(use_grad)

        batched_data = process_batch_data(batched_data, None)
        # Generate embeddings
        atomic_numbers = batched_data["atomic_numbers"]
        padding_mask = ~batched_data["atom_masks"]
        pos = batched_data["pos"]
        batched_data["pos"] = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),
            999.0,
            batched_data["pos"].float(),
        )
        bsz, L = batched_data["pos"].shape[:2]
        # with torch.cuda.amp.autocast(enabled=self.global_cfg.use_fp16_backbone):
        with nullcontext():
            # Handle periodic boundary conditions
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pbc_expand_batched = self.cell_expander.expand_includeself(
                    pos,
                    None,
                    batched_data["pbc"],
                    batched_data["num_atoms"],
                    atomic_numbers,
                    batched_data["cell"],
                    neighbors_radius=(
                        self.kwargs["max_neighbors"],
                        self.kwargs["pbc_max_radius"],
                    ),
                    use_local_attention=False,  # use_local_attention,
                    use_grad=use_grad,
                    padding_mask=padding_mask,
                )
                # dist: B*tgt_len*src_len
                
                pbc_expand_batched["expand_pos"][
                    pbc_expand_batched["expand_mask"]
                ] = 999  # set expand node pos padding to 9999


            else:
                pbc_expand_batched = None


            # boo_fea = self.BOO_feature(pos,expand_pos,local_attention_weight)
            # token_embedding_tgt = self.embedding_tgt(token_id)
            # token_embedding_src = self.embedding_src(torch.gather(token_id,dim = 1,index = outcell_index))
            # dist_rbf = self.dit_gbf(dist,batched_data["node_type_edge"])
            # sigma,co = torch.chunk(self.linear_sigmaco(torch.cat([
            #                 token_embedding_src.unsqueeze(dim = 1).repeat(1,tgt_len,1,1),
            #                 token_embedding_tgt.unsqueeze(dim = 2).repeat(1,1,src_len,1),
            #                 dist_rbf],dim = -1
            #                 )),dim = -1,chunks=2)

            # token_embedding = self.electron_density(
            #     pos,
            #     rji = -pos.unsqueeze(dim = 2)+pbc_expand_batched["expand_pos"].unsqueeze(dim = 1),
            #     sigma = sigma,
            #     co = co,
            #     neighbor_mask = local_attention_weight>1e-5)
            # token_embedding[:,:,0] = token_embedding[:,:,0] + self.embedding(token_id)
            # print(atomic_numbers.shape,batched_data["multiplicity"].squeeze(dim = -1).shape)
            token_embedding = self.embedding(atomic_numbers) + \
                self.embedding_charge(torch.clip(batched_data["charge"],-10,10)+10) + \
                    self.embedding_multiplicity(torch.clip(batched_data["multiplicity"],0,20))

            if self.invariant_encoder is not None:
                token_embedding = self.invariant_encoder(
                    token_embedding, # keep update
                    token_embedding.clone(),  # condition
                    pos,
                    atomic_numbers,
                    padding_mask,
                    batched_data,
                    pbc_expand_batched=pbc_expand_batched,
                )
                # print("layer: ",j,torch.mean(attn),torch.min(attn),torch.max(attn))
                # node_vec_features = node_vec_features * (node_embedding_ef.unsqueeze(dim = -2))
            token_embedding = self.embed_proj(token_embedding)

            # Forward through decoder
            (
                node_features,
                node_vec_features,
                node_irreps,
                node_irreps_his,
            ) = self.decoder(
                batched_data,
                token_embedding,
                None,
                padding_mask,
                pbc_expand_batched=pbc_expand_batched,
                return_node_irreps=True,
            )

        # flatten the node features from num batchs times num nodes to num nodes (to pyG style ), note that nodes are padded
        (
            node_features_flatten,
            node_vec_features_flatten,
            node_irreps_flatten,
            node_irreps_his_flatten,
        ) = self.flatten_node_features(
            node_features,
            node_vec_features,
            node_irreps,
            node_irreps_his,
            ~padding_mask,
        )

        return {
            "node_irrepsBxN": node_irreps,
            "node_featuresBxN": node_features,
            "node_vec_featuresBxN": node_vec_features,
            "data": batched_data,
            "node_irreps": node_irreps_flatten,
            "node_features": node_features_flatten,
            "node_vec_features": node_vec_features_flatten,
            "node_irreps_his": node_irreps_his_flatten,
        }

    def flatten_node_features(
        self,
        node_features,
        node_vec_features,
        node_irreps,
        node_irreps_his,
        padding_mask,
    ):
        flat_node_irreps = node_irreps.view(
            -1, node_irreps.size(-2), node_irreps.size(-1)
        )
        flat_node_irreps_his = node_irreps_his.view(
            -1, node_irreps_his.size(-2), node_irreps_his.size(-1)
        )
        flat_node_features = node_features.view(-1, node_features.size(-1))  # [B*N, D]
        flat_node_vec_features = node_vec_features.view(
            -1, node_vec_features.size(-2), node_vec_features.size(-1)
        )  # [B*N, D_vec]
        flat_mask = padding_mask.view(-1)  # [B*N]
        # Use the mask to filter out padded nodes
        valid_node_irreps = flat_node_irreps[flat_mask]  # [sum(valid_nodes), D]
        valid_node_irreps_his = flat_node_irreps_his[flat_mask]  # [sum(valid_nodes), D]
        valid_node_features = flat_node_features[flat_mask]  # [sum(valid_nodes), D]
        valid_node_vec_features = flat_node_vec_features[flat_mask]
        return (
            valid_node_features,
            valid_node_vec_features,
            valid_node_irreps,
            valid_node_irreps_his,
        )

    def forward(
        self,
        data: torch_geometric.data.Batch,
        node_embedding=None,
        # aa_mask=None,
        # padding_mask=None,
        *args,
        **kwargs,
    ):
        """
        Main forward pass of the model.
        """
        # PSM handles preprocessing internally
        return self.forward_fn(data, token_embedding=node_embedding)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Returns parameters that should not use weight decay.
        """
        return no_weight_decay(self)
        # return no_weight_decay

    def test_equivariant(self, original_data):
        # assume batch size is 1
        assert (
            original_data.batch.max() == 0
        ), "batch size must be 1 for test_equivariant"
        self.eval()  # this is very important
        data_2 = original_data.clone().cpu()
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_2, matrix, inv_matrix = transform(data_2)
        data_2 = data_2.to(original_data.pos.device)
        data_list = [original_data, data_2]
        data_list.ptr = torch.tensor(
            [
                0,
                original_data.pos.size(0),
                original_data.pos.size(0) + data_2.pos.size(0),
            ],
            device=original_data.pos.device,
        )
        results = self.compiled_forward(data_list)
        combined_node_features = results["node_features"]
        # split the node features into two parts
        node_features_1 = combined_node_features[: original_data.pos.size(0)]
        node_features_2 = combined_node_features[original_data.pos.size(0) :]

        assert node_features_1.allclose(
            node_features_2, rtol=1e-2, atol=1e-2
        ), "node features are not equivariant"

        node_vec_features_1 = results["node_vec_features"][: original_data.pos.size(0)]
        node_vec_features_2 = results["node_vec_features"][original_data.pos.size(0) :]
        # rotate the node vec features
        node_vec_features_1 = torch.einsum(
            "bsd, sj -> bjd", node_vec_features_1, matrix.to(node_vec_features_1.device)
        )
        assert node_vec_features_1.allclose(
            node_vec_features_2, rtol=1e-2, atol=1e-2
        ), "node vec features are not equivariant"

class E2FormerHeadBase(nn.Module):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        # self.molecular_graph_cfg = backbone.molecular_graph_cfg
        # self.gnn_cfg = backbone.gnn_cfg
        # self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("E2Former_easy_energy_head")
class E2FormerEasyEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return {"energy": energy_output}


@registry.register_model("E2Former_easy_force_head")
class E2FormerEasyForceHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init()

    def compiled_forward(
        self, node_features, node_vec_features, data
    ):
        # get force direction from node vector features
        force_direction = self.linear(node_vec_features).squeeze(-1)  # (num_nodes, 3)

        # get output force
        return force_direction

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            node_features=emb["node_features"],
            node_vec_features=emb["node_vec_features"],
            data=emb["data"],
        )

        return {"forces": force_output}


@registry.register_model("E2Former_grad_energy_force_head")
class E2FormerGradEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(), data.pos, create_graph=self.training
            )[0]
        )

        return {"energy": energy_output, "forces": forces_output}

