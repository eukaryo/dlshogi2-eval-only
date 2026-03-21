from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ._compat import require_cshogi
from .board_io import load_board
from .features import FEATURES_NUM, legal_moves_and_labels, make_input_features
from .loader import load_model_from_checkpoint
from .model import PolicyValueNetwork


@dataclass
class DensePV:
    policy_logits: torch.Tensor  # [MOVE_LABELS_NUM] on CPU
    value_logit: torch.Tensor  # scalar on CPU

    @property
    def value_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.value_logit)

    def to_jsonable(self, *, include_dense_policy: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value_logit": float(self.value_logit.item()),
            "value_prob": float(self.value_prob.item()),
            "policy_size": int(self.policy_logits.numel()),
        }
        if include_dense_policy:
            payload["policy_logits"] = self.policy_logits.tolist()
        return payload


@dataclass
class LegalPV:
    moves_usi: list[str]
    move_labels: torch.Tensor  # [num_legal] on CPU
    logits: torch.Tensor  # [num_legal] on CPU
    probs: torch.Tensor  # [num_legal] on CPU
    value_logit: torch.Tensor  # scalar on CPU

    @property
    def value_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.value_logit)

    def topk(self, k: int) -> dict[str, Any]:
        if self.probs.numel() == 0:
            return {"moves": [], "indices": [], "probs": [], "logits": []}
        k = min(k, self.probs.numel())
        values, indices = torch.topk(self.probs, k=k)
        idx = indices.tolist()
        return {
            "moves": [self.moves_usi[i] for i in idx],
            "indices": idx,
            "probs": values.tolist(),
            "logits": self.logits[indices].tolist(),
        }

    def to_jsonable(self, *, topk: int | None = None, include_all_legal: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value_logit": float(self.value_logit.item()),
            "value_prob": float(self.value_prob.item()),
            "num_legal_moves": int(self.probs.numel()),
        }
        if topk is not None:
            payload["topk"] = self.topk(topk)
        if include_all_legal:
            payload["legal_moves"] = [
                {
                    "usi": usi,
                    "label": int(label),
                    "logit": float(logit),
                    "prob": float(prob),
                }
                for usi, label, logit, prob in zip(
                    self.moves_usi,
                    self.move_labels.tolist(),
                    self.logits.tolist(),
                    self.probs.tolist(),
                )
            ]
        return payload


class PolicyValueEvaluator:
    def __init__(self, model: PolicyValueNetwork, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str | torch.device = "cpu",
        strict: bool = True,
        blocks: int = 10,
        channels: int = 192,
        fcl: int = 256,
    ) -> "PolicyValueEvaluator":
        model = load_model_from_checkpoint(
            checkpoint_path,
            device=device,
            strict=strict,
            blocks=blocks,
            channels=channels,
            fcl=fcl,
        )
        return cls(model, device=device)

    def encode_board(self, board: Any) -> torch.Tensor:
        np_features = make_input_features(board)
        return torch.from_numpy(np_features).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_dense_from_tensor(self, x: torch.Tensor) -> DensePV:
        x = x.to(self.device)
        policy_logits, value_logits = self.model(x)
        dense_policy = policy_logits.reshape(-1).detach().cpu()
        value_logit = value_logits.reshape(-1)[0].detach().cpu()
        return DensePV(policy_logits=dense_policy, value_logit=value_logit)

    @torch.inference_mode()
    def predict_dense(self, board: Any) -> DensePV:
        x = self.encode_board(board)
        return self.predict_dense_from_tensor(x)

    @torch.inference_mode()
    def predict_legal(
        self,
        board: Any,
        *,
        temperature: float = 1.0,
        dense: DensePV | None = None,
    ) -> LegalPV:
        cshogi = require_cshogi()
        if dense is None:
            dense = self.predict_dense(board)
        moves, labels = legal_moves_and_labels(board)
        moves_usi = [cshogi.move_to_usi(move) for move in moves]

        if len(labels) == 0:
            empty = torch.empty((0,), dtype=torch.float32)
            empty_i64 = torch.empty((0,), dtype=torch.int64)
            return LegalPV(
                moves_usi=[],
                move_labels=empty_i64,
                logits=empty,
                probs=empty,
                value_logit=dense.value_logit,
            )

        label_t = torch.from_numpy(labels.astype(np.int64, copy=False))
        legal_logits = dense.policy_logits.index_select(0, label_t)

        if temperature <= 0:
            probs = torch.zeros_like(legal_logits)
            best_idx = int(torch.argmax(legal_logits).item())
            probs[best_idx] = 1.0
        else:
            probs = torch.softmax(legal_logits / temperature, dim=0)

        return LegalPV(
            moves_usi=moves_usi,
            move_labels=label_t,
            logits=legal_logits,
            probs=probs,
            value_logit=dense.value_logit,
        )

    def predict_from_position(
        self,
        *,
        position: str | None = None,
        sfen: str | None = None,
        temperature: float = 1.0,
    ) -> tuple[DensePV, LegalPV]:
        board = load_board(position=position, sfen=sfen)
        dense = self.predict_dense(board)
        legal = self.predict_legal(board, temperature=temperature, dense=dense)
        return dense, legal
