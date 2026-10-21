"""Variable-degree categorical actor-critic policy and versioned checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl_live import OBSERVATION_SCHEMA_VERSION, POLICY_CHECKPOINT_VERSION
from marl_live.config import FEATURE_DIM, FEATURE_NAMES


class CandidateActorCritic(nn.Module):
    """Shared scoring of candidate next hops, independent of node IDs/degree."""
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features):
        if candidate_features.ndim != 2 or candidate_features.shape[1] != FEATURE_DIM:
            raise ValueError(f"candidate_features must have shape [C,{FEATURE_DIM}]")
        if candidate_features.shape[0] == 0:
            raise ValueError("at least one candidate next hop is required")
        logits = self.scorer(candidate_features).squeeze(-1)
        pooled = candidate_features.mean(dim=0, keepdim=True)
        value = self.value(pooled).squeeze()
        return logits, value


class RoutingPolicy:
    def __init__(self, hidden_dim=64, device="cpu"):
        self.device = torch.device(device)
        self.model = CandidateActorCritic(hidden_dim=hidden_dim).to(self.device)
        self.hidden_dim = int(hidden_dim)
        self.model.eval()

    def distribution(self, observation, stochastic=False):
        x = torch.as_tensor(observation["candidate_features"], dtype=torch.float32,
                            device=self.device)
        mask = torch.as_tensor(observation["action_mask"], dtype=torch.bool,
                               device=self.device)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or mask.shape != (x.shape[0],):
            raise ValueError("observation schema does not match candidate-link-v1")
        if not bool(mask.any()):
            raise ValueError("no valid next-hop actions")
        logits, value = self.model(x)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def select_action(self, observation, stochastic=False):
        self.model.eval()
        dist, _ = self.distribution(observation)
        action = dist.sample() if stochastic else dist.probs.argmax()
        return int(action.item())

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": POLICY_CHECKPOINT_VERSION,
            "observation_schema": OBSERVATION_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "hidden_dim": self.hidden_dim,
            "policy_state_dict": self.model.state_dict(),
            "metadata": metadata,
        }
        torch.save(payload, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, device="cpu"):
        # Older run manifests may have saved torch.__version__ as TorchVersion
        # (a str subclass). Explicitly allowlist that harmless metadata type;
        # do not disable weights_only for arbitrary checkpoint paths.
        try:
            from torch.torch_version import TorchVersion
            safe_globals = [TorchVersion]
        except ImportError:
            safe_globals = []
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("checkpoint_version") != POLICY_CHECKPOINT_VERSION:
            raise ValueError("unsupported MARL live policy checkpoint version")
        if checkpoint.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("checkpoint observation schema is incompatible")
        if checkpoint.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("checkpoint candidate feature ordering is incompatible")
        policy = cls(hidden_dim=int(checkpoint["hidden_dim"]), device=device)
        policy.model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        policy.model.eval()
        policy.metadata = checkpoint.get("metadata", {})
        return policy
