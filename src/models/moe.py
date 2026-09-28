"""
src/models/moe.py

Sparse Mixture of Experts (MoE) Module for Scalable Spatiotemporal Diffusion.
Mathematical Formulation:
  1. Top-k Routing:
     Given token or feature representations x in R^[N, D]:
       h(x) = W_gate * x + noise
       p(x) = Softmax(TopK(h(x), k))
  2. Sparse Dispatch & Combine:
     y = sum_{i in TopK} p_i(x) * Expert_i(x)
  3. Load-Balancing Auxiliary Loss (Switch / GShard):
     L_aux = E * sum_{e=1}^E f_e * P_e
     where:
       f_e = (1 / N) * sum_t I(expert e selected for token t)
       P_e = (1 / N) * sum_t p_{t, e}
  4. Diagnostics:
     Routing entropy H = -sum_{e=1}^E P_e * log(P_e + eps)
     Tracks expert utilization, load imbalance, and dead-expert detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_diffusion import TimeConditionedConvNeXtBlock


class TopKRouter(nn.Module):
    """
    Top-k Sparse Router with auxiliary load-balancing loss and entropy tracking.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        noise_std: float = 0.0,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts, f"top_k ({top_k}) must be between 1 and num_experts ({num_experts})"
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Routes tokens to top-k experts.

        Args:
            x: Tensor of shape [N, D] where N = total tokens / locations.
        Returns:
            topk_weights: [N, top_k] normalized dispatch weights.
            topk_indices: [N, top_k] expert indices for each token.
            aux_loss: scalar auxiliary load-balancing loss.
            metrics: dict containing routing entropy, expert utilization, load balance.
        """
        num_tokens, feat_dim = x.shape
        logits = self.gate(x)  # [N, num_experts]

        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * self.noise_std
            gating_logits = logits + noise
        else:
            gating_logits = logits

        # Full softmax over all experts for auxiliary loss
        full_probs = F.softmax(gating_logits, dim=-1)  # [N, num_experts]

        # Top-k selection
        topk_scores, topk_indices = torch.topk(full_probs, self.top_k, dim=-1)  # [N, top_k]

        # Re-normalize top-k weights so they sum to 1.0 per token
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Auxiliary load balancing loss (Switch Transformer / GShard formulation):
        # f_e: fraction of tokens dispatched to expert e
        # P_e: average probability assigned to expert e across all tokens
        mask = torch.zeros_like(full_probs).scatter_(-1, topk_indices, 1.0)
        f_e = mask.mean(dim=0)  # [num_experts]
        P_e = full_probs.mean(dim=0)  # [num_experts]

        # L_aux = num_experts * sum_e (f_e * P_e)
        aux_loss = self.aux_loss_weight * self.num_experts * torch.sum(f_e * P_e)

        # Routing entropy: H = - sum_e P_e * log(P_e + eps)
        eps = 1e-8
        entropy = -torch.sum(P_e * torch.log(P_e + eps))
        max_possible_entropy = torch.log(torch.tensor(float(self.num_experts), device=x.device))
        normalized_entropy = entropy / (max_possible_entropy + eps)

        # Dead expert detection (experts receiving < 1% of expected load)
        expected_load = 1.0 / self.num_experts
        dead_experts = (f_e < 0.01 * expected_load).sum().item()

        metrics = {
            "entropy": entropy.item(),
            "normalized_entropy": normalized_entropy.item(),
            "expert_frequencies": f_e.detach().cpu().tolist(),
            "expert_probabilities": P_e.detach().cpu().tolist(),
            "dead_experts": int(dead_experts),
            "max_load_fraction": f_e.max().item(),
            "min_load_fraction": f_e.min().item(),
        }

        return topk_weights, topk_indices, aux_loss, metrics


class MoETimeConditionedConvNeXtBlock(nn.Module):
    """
    Time-Conditioned ConvNeXt Block with Sparse Mixture of Experts.
    Replaces dense pointwise FFN with E experts routed by TopKRouter.
    """

    def __init__(
        self,
        dim: int,
        time_emb_dim: int,
        expansion: int = 2,
        num_groups: int = 8,
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        assert dim % num_groups == 0
        self.dim = dim
        self.time_emb_dim = time_emb_dim
        self.expansion = expansion
        self.num_groups = num_groups
        self.num_experts = num_experts
        self.top_k = top_k

        # Shared depthwise convolution and GroupNorm
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.time_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim * 2),
        )

        # Router operating on spatially-pooled or spatial tokens
        self.router = TopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
        )

        # List of E pointwise MLP experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * expansion, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(dim * expansion, dim, kernel_size=1),
            )
            for _ in range(num_experts)
        ])

        # Track last auxiliary loss and routing metrics
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_metrics: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse top-k expert dispatch.
        Args:
            x: [B_total, dim, H, W]
            time_emb: [B_total, time_emb_dim]
        Returns:
            out: [B_total, dim, H, W]
        """
        res = x
        b_total, dim, h, w = x.shape

        out = self.dwconv(x)
        out = self.norm(out)

        # Time modulation: scale and shift
        scale_shift = self.time_proj(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        out = out * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        # Compute routing representation per sample (spatially pooled token)
        # This keeps spatial coherence within each weather field while routing samples/leads
        token_rep = F.adaptive_avg_pool2d(out, 1).view(b_total, dim)  # [B_total, dim]

        topk_weights, topk_indices, aux_loss, metrics = self.router(token_rep)
        self.last_aux_loss = aux_loss
        self.last_metrics = metrics

        # Dispatch and combine outputs
        combined = torch.zeros_like(out)
        for k_idx in range(self.top_k):
            indices_k = topk_indices[:, k_idx]  # [B_total]
            weights_k = topk_weights[:, k_idx].view(b_total, 1, 1, 1)  # [B_total, 1, 1, 1]

            for expert_id, expert in enumerate(self.experts):
                expert_mask = (indices_k == expert_id)
                if expert_mask.any():
                    expert_in = out[expert_mask]
                    expert_out = expert(expert_in)
                    combined[expert_mask] = combined[expert_mask] + weights_k[expert_mask] * expert_out

        return res + combined

    def get_active_parameters(self) -> int:
        """Computes active parameters during forward pass with top-k routing."""
        shared_params = (
            sum(p.numel() for p in self.dwconv.parameters())
            + sum(p.numel() for p in self.norm.parameters())
            + sum(p.numel() for p in self.time_proj.parameters())
            + sum(p.numel() for p in self.router.parameters())
        )
        expert_params = sum(p.numel() for p in self.experts[0].parameters()) * self.top_k
        return shared_params + expert_params
