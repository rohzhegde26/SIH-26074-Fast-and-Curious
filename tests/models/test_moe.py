"""
tests/models/test_moe.py

TDD Unit Tests for MoE Sparse Routing, Load Balancing, and Expert Dispatch.
"""

import pytest
import torch

from src.models.moe import (
    MoETimeConditionedConvNeXtBlock,
    TopKRouter,
)


def test_topk_router_shapes_and_probabilities():
    """Verify TopKRouter produces valid normalized weights and correct shapes."""
    num_tokens = 32
    feat_dim = 64
    num_experts = 4
    top_k = 2

    router = TopKRouter(
        dim=feat_dim,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_weight=0.01,
    )

    x = torch.randn(num_tokens, feat_dim)
    weights, indices, aux_loss, metrics = router(x)

    assert weights.shape == (num_tokens, top_k)
    assert indices.shape == (num_tokens, top_k)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0
    assert torch.isfinite(aux_loss)

    # Weights must sum to 1.0 per token
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)

    # Check metrics
    assert "entropy" in metrics
    assert "normalized_entropy" in metrics
    assert "expert_frequencies" in metrics
    assert len(metrics["expert_frequencies"]) == num_experts


def test_moe_convnext_block_forward_backward():
    """Verify MoETimeConditionedConvNeXtBlock forward/backward pass and parameter accounting."""
    b = 4
    dim = 64
    h, w = 10, 10
    time_emb_dim = 32
    num_experts = 4
    top_k = 1

    block = MoETimeConditionedConvNeXtBlock(
        dim=dim,
        time_emb_dim=time_emb_dim,
        num_groups=8,
        num_experts=num_experts,
        top_k=top_k,
    )

    x = torch.randn(b, dim, h, w, requires_grad=True)
    time_emb = torch.randn(b, time_emb_dim)

    out = block(x, time_emb)
    assert out.shape == (b, dim, h, w)
    assert torch.isfinite(out).all()

    # Verify active parameters is strictly less than total parameters
    total_p = sum(p.numel() for p in block.parameters())
    active_p = block.get_active_parameters()
    assert active_p < total_p

    # Backward pass
    loss = out.sum() + block.last_aux_loss
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("top_k", [1, 2])
def test_moe_block_different_top_k(top_k: int):
    """Verify MoE block operates correctly across top_k in {1, 2}."""
    b = 2
    dim = 32
    block = MoETimeConditionedConvNeXtBlock(
        dim=dim,
        time_emb_dim=16,
        num_groups=4,
        num_experts=4,
        top_k=top_k,
    )
    x = torch.randn(b, dim, 8, 8)
    time_emb = torch.randn(b, 16)
    out = block(x, time_emb)
    assert out.shape == (b, dim, 8, 8)
    assert torch.isfinite(out).all()
