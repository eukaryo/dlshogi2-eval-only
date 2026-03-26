from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ._compat import require_cshogi
from .board_io import load_board
from .features import legal_moves_and_labels, make_input_features
from .loader import load_model_from_checkpoint
from .model import PolicyValueNetwork


def _normalize_precision_name(value: str | torch.dtype | None) -> str:
    if value is None:
        return "fp32"
    if isinstance(value, torch.dtype):
        mapping = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }
        if value not in mapping:
            raise ValueError(f"unsupported torch dtype: {value}")
        return mapping[value]
    name = str(value).strip().lower()
    aliases = {
        "float32": "fp32",
        "f32": "fp32",
        "fp32": "fp32",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "bf16": "bf16",
    }
    if name not in aliases:
        raise ValueError(f"unsupported precision: {value!r}")
    return aliases[name]


def precision_name_to_torch_dtype(value: str | torch.dtype | None) -> torch.dtype:
    name = _normalize_precision_name(value)
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise AssertionError(f"unreachable precision name: {name}")


@dataclass(frozen=True)
class BackendSpec:
    kind: str
    device: str
    precision: str
    use_autocast: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device": self.device,
            "precision": self.precision,
            "use_autocast": self.use_autocast,
        }


@dataclass
class DensePV:
    policy_logits: torch.Tensor  # [MOVE_LABELS_NUM] on CPU / float32
    value_logit: torch.Tensor  # scalar on CPU / float32

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
    logits: torch.Tensor  # [num_legal] on CPU / float32
    probs: torch.Tensor  # [num_legal] on CPU / float32
    value_logit: torch.Tensor  # scalar on CPU / float32

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

    def to_jsonable(
        self,
        *,
        topk: int | None = None,
        include_all_legal: bool = False,
    ) -> dict[str, Any]:
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


def make_backend_payload(
    *,
    backend: BackendSpec | dict[str, Any],
    dense: DensePV,
    legal: LegalPV,
    include_dense_policy: bool = False,
    topk: int | None = None,
    include_all_legal: bool = False,
) -> dict[str, Any]:
    backend_payload = backend.to_jsonable() if isinstance(backend, BackendSpec) else dict(backend)
    return {
        "backend": backend_payload,
        "dense": dense.to_jsonable(include_dense_policy=include_dense_policy),
        "legal": legal.to_jsonable(topk=topk, include_all_legal=include_all_legal),
    }


def dense_from_output_arrays(
    outputs: dict[str, np.ndarray] | dict[str, torch.Tensor],
) -> DensePV:
    if "policy_logits" not in outputs:
        raise KeyError("outputs must contain 'policy_logits'")
    if "value_logit" not in outputs:
        raise KeyError("outputs must contain 'value_logit'")

    policy_arr = outputs["policy_logits"]
    value_arr = outputs["value_logit"]

    if isinstance(policy_arr, torch.Tensor):
        policy_t = policy_arr.detach().float().cpu()
    else:
        policy_t = torch.from_numpy(np.asarray(policy_arr)).float().cpu()

    if isinstance(value_arr, torch.Tensor):
        value_t = value_arr.detach().float().cpu()
    else:
        value_t = torch.from_numpy(np.asarray(value_arr)).float().cpu()

    dense_policy = policy_t.reshape(-1).contiguous()
    value_logit = value_t.reshape(-1)[0].contiguous()
    return DensePV(policy_logits=dense_policy, value_logit=value_logit)


def raw_output_arrays_from_dense(dense: DensePV) -> dict[str, np.ndarray]:
    return {
        "policy_logits": dense.policy_logits.detach().float().cpu().reshape(1, -1).numpy(),
        "value_logit": dense.value_logit.detach().float().cpu().reshape(1, 1).numpy(),
    }


def legal_from_dense(
    board: Any,
    dense: DensePV,
    *,
    temperature: float = 1.0,
) -> LegalPV:
    cshogi = require_cshogi()
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
    legal_logits = dense.policy_logits.index_select(0, label_t).float()
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


class PolicyValueEvaluator:
    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        device: str | torch.device = "cpu",
        precision: str | torch.dtype | None = None,
        use_autocast: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.precision = _normalize_precision_name(precision)
        self.compute_dtype = precision_name_to_torch_dtype(self.precision)
        self.use_autocast = bool(use_autocast)
        self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()

    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            kind="pytorch",
            device=str(self.device),
            precision=self.precision,
            use_autocast=self.use_autocast,
        )

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
        precision: str | torch.dtype | None = None,
        use_autocast: bool = False,
    ) -> "PolicyValueEvaluator":
        model = load_model_from_checkpoint(
            checkpoint_path,
            device=device,
            strict=strict,
            blocks=blocks,
            channels=channels,
            fcl=fcl,
        )
        return cls(model, device=device, precision=precision, use_autocast=use_autocast)

    def encode_board(self, board: Any) -> torch.Tensor:
        np_features = make_input_features(board)
        return (
            torch.from_numpy(np_features)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.compute_dtype)
        )

    def _autocast_context(self):
        if not self.use_autocast:
            return contextlib.nullcontext()
        if self.precision == "fp32":
            return contextlib.nullcontext()
        if self.device.type not in {"cuda", "cpu"}:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)

    @torch.inference_mode()
    def predict_raw_outputs_from_tensor(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        x = x.to(device=self.device, dtype=self.compute_dtype)
        with self._autocast_context():
            policy_logits, value_logits = self.model(x)
        return {
            "policy_logits": policy_logits.detach().float().cpu().contiguous().numpy(),
            "value_logit": value_logits.detach().float().cpu().contiguous().numpy(),
        }

    @torch.inference_mode()
    def predict_dense_from_tensor(self, x: torch.Tensor) -> DensePV:
        outputs = self.predict_raw_outputs_from_tensor(x)
        return dense_from_output_arrays(outputs)

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
        if dense is None:
            dense = self.predict_dense(board)
        return legal_from_dense(board, dense, temperature=temperature)

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
