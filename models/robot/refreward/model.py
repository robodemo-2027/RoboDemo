"""Appendix O architecture. Backbones are deliberately outside this module."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .features import CandidateFeatures, InitialFeatures

D, HEADS, LATENTS = 256, 8, 32


def linear(a, b):
    return nn.Linear(a, b, bias=False)


def norm():
    return nn.LayerNorm(D, elementwise_affine=False)


def ffn():
    return nn.Sequential(linear(D, 4 * D), nn.GELU(), linear(4 * D, D))


def position_encoding(position: Tensor) -> Tensor:
    frequency = torch.exp(torch.arange(0, D, 2, device=position.device).float() * (-math.log(10000) / D))
    phase = position.float().unsqueeze(-1) * frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


def attention(q, k, v, mask, bias=None):
    def split(x):
        return x.reshape(x.shape[0], -1, HEADS, D // HEADS).transpose(1, 2)
    q, k, v = map(split, (q, k, v))
    logits = q.float() @ k.float().transpose(-2, -1) / math.sqrt(D // HEADS)
    if bias is not None:
        logits = logits + bias.float()
    valid = mask[:, None, None, :]
    # Avoid softmax(NaN) when an entire sample is masked.
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = logits.softmax(-1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    out = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(q.shape[0], -1, D)
    return out, weights


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = linear(D, 3 * D)
        self.output = linear(D, D)

    def forward(self, x, mask, bias=None):
        q, k, v = self.qkv(x).chunk(3, -1)
        return self.output(attention(q, k, v, mask, bias)[0])


class SpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = norm(), norm()
        self.attention, self.ffn = SelfAttention(), ffn()

    def forward(self, x, mask, bias):
        x = x + self.attention(self.ln1(x), mask, bias)
        return (x + self.ffn(self.ln2(x))) * mask.unsqueeze(-1)


class RewardBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = norm(), norm(), norm()
        self.query, self.output = linear(D, D), linear(D, D)
        self.attention, self.ffn = SelfAttention(), ffn()

    def forward(self, r, k, v, mask):
        x, weights = attention(self.query(self.ln1(r)), k, v, mask)
        r = r + self.output(x)
        valid = torch.ones(r.shape[:2], device=r.device, dtype=torch.bool)
        r = r + self.attention(self.ln2(r), valid)
        return r + self.ffn(self.ln3(r)), weights


@dataclass
class KV:
    k: Tensor
    v: Tensor
    mask: Tensor


@dataclass
class RewardOutput:
    reward: Tensor
    gate_logits: Tensor
    factor_logits: Tensor
    evidence: Tensor
    required_evidence_valid: Tensor


class RefReward(nn.Module):
    def __init__(self, *, dino_dim=768):
        super().__init__()
        if dino_dim not in (768, 1280):
            raise ValueError("explicit B/H+ DINO width required")
        self.dino_dim = dino_dim
        self.projections = nn.ModuleDict({
            "g": linear(2048, D), "z": linear(2048, D), "T": linear(2048, D),
            "F": linear(dino_dim, D), "C": linear(24, D), "P": linear(4, D),
        })
        self.relative_bias = nn.Sequential(linear(4, 32), nn.GELU(), linear(32, HEADS))
        self.spatial = nn.ModuleList([SpatialBlock() for _ in range(2)])
        self.evidence_norm = norm()
        self.key, self.value = linear(D, D), linear(D, D)
        self.blocks = nn.ModuleList([RewardBlock() for _ in range(4)])
        self.latents = nn.Parameter(torch.randn(LATENTS, D) * 0.02)
        self.roles_modalities = nn.Parameter(torch.randn(7, D) * 0.02)
        self.head = nn.Sequential(linear(D, D), nn.GELU(), linear(D, 9))
        self.ordinal_parameters = nn.Parameter(torch.tensor([-1.5, 0.0, 0.0, 0.0]))

    def thresholds(self):
        return torch.cat((
            self.ordinal_parameters[:1],
            self.ordinal_parameters[:1] + F.softplus(self.ordinal_parameters[1:]).cumsum(0)
        ))

    def encode(self, features: InitialFeatures, *, candidate=False) -> KV:
        if features.dino_dim != self.dino_dim:
            raise ValueError("DINO feature/model width mismatch; B/H+ caches cannot be mixed")
        if candidate != isinstance(features, CandidateFeatures):
            raise TypeError("query inputs must be initial-only; candidates require process fields")
        p = self.projections
        xy = features.P[..., :2]
        delta = xy[:, :, None, :] - xy[:, None, :, :]
        bias = self.relative_bias(torch.cat((delta, delta.abs()), -1)).permute(0, 3, 1, 2)
        x = p["F"](F.normalize(features.F.float(), dim=-1).to(features.F.dtype))
        x = x + p["C"](features.C) + p["P"](features.P)
        x = x * features.spatial_mask.unsqueeze(-1)
        for block in self.spatial:
            x = block(x, features.spatial_mask, bias)
        m = self.roles_modalities
        parts = [
            p["g"](features.g).unsqueeze(1) + m[0],
            p["z"](features.z).unsqueeze(1) + m[1],
            p["T"](features.T) + position_encoding(features.text_position).to(x.dtype) + m[2],
            x + m[3],
        ]
        masks = [features.global_mask, features.text_mask, features.spatial_mask]
        if candidate:
            parts.append(p["z"](features.U) + position_encoding(features.process_time).to(x.dtype) + m[4])
            masks.append(features.process_mask)
        evidence = torch.cat(parts, 1) + m[6 if candidate else 5]
        mask = torch.cat(masks, 1)
        evidence = self.evidence_norm(evidence) * mask.unsqueeze(-1)
        return KV(self.key(evidence), self.value(evidence), mask)

    def score_kv(self, query: KV, candidate: KV) -> RewardOutput:
        batch = candidate.k.shape[0]
        if query.k.shape[1] != 290 or candidate.k.shape[1] != 322:
            raise ValueError("invalid query/candidate evidence lengths")
        if query.k.shape[0] not in (1, batch):
            raise ValueError("query batch must be singleton or match candidates")
        k = torch.cat((query.k.expand(batch, -1, -1), candidate.k), 1)
        v = torch.cat((query.v.expand(batch, -1, -1), candidate.v), 1)
        mask = torch.cat((query.mask.expand(batch, -1), candidate.mask), 1)
        r = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            r, weights = block(r, k, v, mask)
        values = self.head(r.mean(1))
        support = weights[..., -32:].mean((1, 2)) * candidate.mask[:, -32:]
        support = support / support.sum(-1, keepdim=True).clamp_min(1e-12)
        def valid_initial(m):
            return m[:, :2].all(-1) & m[:, 2:34].any(-1) & m[:, 34:290].any(-1)
        valid = valid_initial(query.mask).expand(batch) & valid_initial(candidate.mask)
        valid = valid & candidate.mask[:, -32:].any(-1)
        return RewardOutput(values[:, 0], values[:, 1:3], values[:, 3:], support, valid)

    def forward(self, query: InitialFeatures, candidate: CandidateFeatures):
        return self.score_kv(self.encode(query), self.encode(candidate, candidate=True))

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
