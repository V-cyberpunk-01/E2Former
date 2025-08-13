
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """
    128 专家，固定 Top-K=8 路由。支持容量裁剪与负载均衡辅助损失。
    主要向量化策略：
      - 先对每个 token 取 TopK（K=8），展开为 (N*K) 的候选对 (tok, expert)
      - 使用专家级循环（E=128）做容量截断，避免 N 或 K 维度的 Python 循环
      - 专家前向为每专家一次 batched GEMM（<= capacity）
      - 聚合时用 index_add_ / scatter_add 完成按 token 汇总
    """
    def __init__(
        self,
        router_fea_dim: int,
        num_experts: int = 128,
        experts = None,
        k: int = 8,
        capacity_factor: float = 1.25,
        aux_loss_type: str = "gshard",   # "gshard" 或 "switch"
        router_temperature: float = 1.0,
        router_noisy: bool = False,
        expert_activation=nn.SiLU,
    ):
        super().__init__()
        assert 1 <= k <= num_experts
        self.router_fea_dim = router_fea_dim
        # self.d_ff = d_ff
        self.E = num_experts
        self.K = k
        self.capacity_factor = capacity_factor
        self.aux_loss_type = aux_loss_type.lower()
        self.router_temperature = router_temperature
        self.router_noisy = router_noisy

        self.router = nn.Linear(router_fea_dim, self.E, bias=False)
        self.experts = experts #nn.ModuleList([ExpertFFN(d_model, d_ff, activation=expert_activation)
                                    #   for _ in range(self.E)])

    @staticmethod
    def _gumbel_noise_like(t):
        u = torch.rand_like(t)
        return -torch.log(-torch.log(u.clamp_min(1e-9)).clamp_min(1e-9))

    def _aux_load_balancing(self, probs: torch.Tensor, assigned_one_hot: torch.Tensor) -> torch.Tensor:
        """
        probs: [N, E] 软概率（未截断）
        assigned_one_hot: [N, E] 0/1，TopK 和 capacity 截断后
        """
        E = probs.size(1)
        importance = probs.sum(dim=0)                   # [E]
        load = assigned_one_hot.sum(dim=0)              # [E]

        if self.aux_loss_type == "gshard":
            imp_n = importance / (importance.sum() + 1e-9)
            load_n = load / (load.sum() + 1e-9)
            aux = E * (imp_n * load_n).sum()
        elif self.aux_loss_type == "switch":
            def cv2(x):
                m = x.mean()
                return ((x - m) ** 2).mean() / (m ** 2 + 1e-9)
            aux = cv2(importance) + cv2(load)
        else:
            raise ValueError("aux_loss_type must be 'gshard' or 'switch'")
        return aux

    def forward(self, 

            node_embedding: torch.Tensor, 
            router_fea: torch.Tensor, 
            return_aux: bool = True):
        """
        node_embedding: [B, N, (lmax+1)**2, C]
        router_fea: [B,  C]
        返回: y: same like node_embedding [B,N, (lmax+1)**2, C], aux_loss: 标量
        训练/推理同一路由，固定 Top-K=8。
        """
        # B, N, C = node_embedding.shape
        # router_fea = node_embedding.reshape(N, C)
        B = router_fea.shape[0]

        # 1) Router softmax 概率
        logits = self.router(router_fea) / self.router_temperature   # [N, E]
        if self.router_noisy and self.training:
            logits = logits + self._gumbel_noise_like(logits)
        probs = F.softmax(logits, dim=-1)                        # [N, E]

        # 2) 固定 Top-K（K=8），得到 (N*K) 的候选对
        topk_val, topk_idx = probs.topk(self.K, dim=-1)          # [N, K]
        # 先按候选权重归一化（便于后续 capacity 掉点再二次归一）
        topk_w = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)  # [N, K]

        tok_idx = torch.arange(B, device=node_embedding.device).unsqueeze(1).expand(B, self.K)  # [N, K]
        flat_tok = tok_idx.reshape(-1)     # [N*K]
        flat_exp = topk_idx.reshape(-1)    # [N*K]
        flat_w   = topk_w.reshape(-1)      # [N*K]

        # 3) 容量限制：每专家最多接收 capacity 个 token（保留权重大的）
        capacity = math.ceil(self.capacity_factor * (B / self.E))

        # 这里按专家循环（E=128），但不按 token 循环；对每专家进行一次索引筛选与排序。
        kept_mask_flat = torch.zeros_like(flat_w, dtype=torch.bool)
        # 记录每个专家保留 token 的“全局 token 索引”，用于 pack
        kept_tok_per_exp = [None] * self.E

        # 为了快：先一次性按 expert 分桶（bincount+分桶索引），减少 where 次数
        # 构造每个 expert 的索引列表（向量化），然后专家级循环内只做排序和切片
        # 排序依据是 flat_w（越大越优先）
        # 注：若 N*K 很大，以下专家级循环在 E=128 下仍然很快
        for e in range(self.E):
            mask_e = (flat_exp == e)
            if not mask_e.any():
                kept_tok_per_exp[e] = torch.empty(0, dtype=torch.long, device=node_embedding.device)
                continue
            idx_e = torch.nonzero(mask_e, as_tuple=False).squeeze(-1)       # 候选对在扁平数组中的位置
            # 选权重 top-capacity
            if idx_e.numel() > capacity:
                # 只对该专家的候选进行局部 topk，避免全局排序开销
                sub_w = flat_w.index_select(0, idx_e)
                top_w, top_pos = torch.topk(sub_w, k=capacity, largest=True, sorted=False)
                kept_local = idx_e.index_select(0, top_pos)
            else:
                kept_local = idx_e
            kept_mask_flat[kept_local] = True
            kept_tok_per_exp[e] = flat_tok.index_select(0, kept_local)

        # 4) 计算每个 token 的最终聚合权重（把被 capacity 裁掉的候选权重置零并二次归一）
        kept_mask = kept_mask_flat.view(B, self.K)            # [N, K]
        kept_w = (topk_w * kept_mask)                         # [N, K]
        kept_w = kept_w / (kept_w.sum(dim=-1, keepdim=True) + 1e-9)  # 二次归一
        kept_w_flat = kept_w.reshape(-1)                      # [N*K]

        # 5) 为每个专家 pack 输入：X_e: [ne, C]，ne<=capacity
        #    用一次性大张量来承载专家输出，后续按 token 索引 scatter 回去
        y_flat =  torch.zeros_like(node_embedding) # x_flat.new_zeros(N, C)
        assigned_one_hot = torch.zeros(B, self.E, device=node_embedding.device, dtype=torch.float32)

        # 仅按专家循环；每次用 index_select 拉取该专家的 token 批次，做一轮 GEMM
        for e in range(self.E):
            tok_e = kept_tok_per_exp[e]            # [ne]
            ne = tok_e.numel()
            if ne == 0:
                continue
            # 记录 assigned 供 aux_loss 统计
            assigned_one_hot.index_put_((tok_e, torch.full((ne,), e, device=node_embedding.device, dtype=torch.long)),
                                        torch.ones(ne, device=node_embedding.device), accumulate=True)

            x_e = node_embedding.index_select(0, tok_e)    # [ne, C]
            out_e = self.experts[e](x_e)           # [ne, C]

            # 取出对应的聚合权重：需要在 (N,K) 的 kept 对应列里抓出这些 (tok, e) 组合的权重
            # 先找到 tok_e 各自对应到其 K 个候选中 where(exp==e) 的列号，再 gather 权重
            # 做法：对 tok_e 批次构造 mask (K 列)，从 topk_idx 抓出 e 的位置
            topk_idx_tok = topk_idx.index_select(0, tok_e)          # [ne, K]
            kept_w_tok   = kept_w.index_select(0, tok_e)            # [ne, K]
            # 每行只会有至多 1 个列等于 e（TopK 中选到 e），有可能 0（被 capacity 裁掉则 kept_w 为 0）
            col_mask = (topk_idx_tok == e)
            # 将每行的 e 列聚合为标量权重： (ne, K) * (ne, K) -> (ne,)
            w_e = (kept_w_tok * col_mask).sum(dim=-1, keepdim=True) # [ne, 1]

            # 聚合到 y_flat：y[tok_e] += w_e * out_e
            y_flat.index_add_(0, tok_e, w_e * out_e)

        # y = y_flat.view(B, T, C)

        aux = self._aux_load_balancing(probs, assigned_one_hot) if return_aux else torch.tensor(0., device=node_embedding.device)
        return y_flat, aux


# moe = MoE(1024,
#                    num_experts=16, k=3,
#                    experts = nn.ModuleList([nn.Linear(1024,1024) for _ in range(16)]),
#                    capacity_factor=1.25,
#                    aux_loss_type="gshard",
#                    router_temperature=1.0,
#                    router_noisy=False) #.cuda()
# node_embedding = torch.randn(2,4,9,1024) #.cuda()
# router_fea = torch.randn(2,1024)
# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#     for i in range(100):
#         y, aux = moe(node_embedding,router_fea)                      # x: [B,T,1024]
# # task_loss = criterion(y, target)
# # loss = task_loss + 0.9 * aux            # router_aux_loss_coef
# # loss.backward()
# # optimizer.step()