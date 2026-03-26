"""Eval-only reference extraction of python-dlshogi2."""

from .features import FEATURES_NUM, MOVE_LABELS_NUM, MOVE_PLANES_NUM
from .infer import (
    BackendSpec,
    DensePV,
    LegalPV,
    PolicyValueEvaluator,
    dense_from_output_arrays,
    legal_from_dense,
    make_backend_payload,
    precision_name_to_torch_dtype,
    raw_output_arrays_from_dense,
)
from .loader import load_model_from_checkpoint
from .model import PolicyValueNetwork
from .openheart_interpret import interpret_openheart_outputs, load_openheart_outputs
from .openheart_package import (
    OpenHeartPackageArtifacts,
    build_openheart_package_id,
    export_openheart_package,
    extract_torch_export_bindings,
    validate_openheart_package_dir,
)

__all__ = [
    "FEATURES_NUM",
    "MOVE_LABELS_NUM",
    "MOVE_PLANES_NUM",
    "BackendSpec",
    "DensePV",
    "LegalPV",
    "PolicyValueEvaluator",
    "PolicyValueNetwork",
    "dense_from_output_arrays",
    "raw_output_arrays_from_dense",
    "legal_from_dense",
    "make_backend_payload",
    "precision_name_to_torch_dtype",
    "load_model_from_checkpoint",
    "load_openheart_outputs",
    "interpret_openheart_outputs",
    "OpenHeartPackageArtifacts",
    "build_openheart_package_id",
    "export_openheart_package",
    "extract_torch_export_bindings",
    "validate_openheart_package_dir",
]

__version__ = "0.2.0"
