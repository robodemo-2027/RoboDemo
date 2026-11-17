"""Explicit development revision; the original model module remains immutable."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import CandidateFeatures
from .model import D, KV, RefReward
from .model import position_encoding


@dataclass
class RevisionKV(KV):
    g: torch.Tensor
    z: torch.Tensor


class RefRewardRevision(RefReward):
    def __init__(self, *, balanced=False, joint_prior=False, dino_dim=1280):
        super().__init__(dino_dim=dino_dim)
        self.balanced, self.joint_prior = balanced, joint_prior
        if joint_prior:
            self.joint_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(20.))))
            nn.init.zeros_(self.head[-1].weight)

    def encode(self, features, *, candidate=False):
        if not self.balanced:
            base = super().encode(features, candidate=candidate)
        else:
            if features.dino_dim != self.dino_dim:
                raise ValueError("feature/model DINO width mismatch")
            if candidate != isinstance(features, CandidateFeatures):
                raise TypeError("queries must be initial-only")
            p = self.projections
            unit = lambda value: (F.normalize(value.float(), dim=-1) * math.sqrt(D)).to(value.dtype)
            xy = features.P[..., :2]
            delta = xy[:, :, None, :] - xy[:, None, :, :]
            bias = self.relative_bias(torch.cat((delta, delta.abs()), -1)).permute(0, 3, 1, 2)
            x = unit(p["F"](F.normalize(features.F.float(), dim=-1).to(features.F.dtype)))
            x = x + .25 * unit(p["C"](features.C)) + .25 * unit(p["P"](features.P))
            x = x * features.spatial_mask.unsqueeze(-1)
            for block in self.spatial:
                x = block(x, features.spatial_mask, bias)
            m = 16 * self.roles_modalities
            parts = [
                unit(p["g"](features.g)).unsqueeze(1) + m[0],
                unit(p["z"](features.z)).unsqueeze(1) + m[1],
                unit(p["T"](features.T)) + .25 * position_encoding(features.text_position).to(x.dtype) + m[2],
                x + m[3],
            ]
            masks = [features.global_mask, features.text_mask, features.spatial_mask]
            if candidate:
                parts.append(unit(p["z"](features.U))
                             + .25 * position_encoding(features.process_time).to(x.dtype) + m[4])
                masks.append(features.process_mask)
            evidence = torch.cat(parts, 1) + m[6 if candidate else 5]
            mask = torch.cat(masks, 1)
            evidence = self.evidence_norm(evidence) * mask.unsqueeze(-1)
            base = KV(self.key(evidence), self.value(evidence), mask)
        return RevisionKV(base.k, base.v, base.mask,
                          F.normalize(features.g.float(), dim=-1),
                          F.normalize(features.z.float(), dim=-1))

    def score_kv(self, query, candidate):
        output = super().score_kv(query, candidate)
        if self.joint_prior:
            if not isinstance(query, RevisionKV) or not isinstance(candidate, RevisionKV):
                raise TypeError("joint-prior cached scoring needs raw normalized globals")
            similarity = (query.z.expand_as(candidate.z) * candidate.z).sum(-1)
            output.reward = output.reward + F.softplus(self.joint_log_scale) * (similarity - .9)
        return output


