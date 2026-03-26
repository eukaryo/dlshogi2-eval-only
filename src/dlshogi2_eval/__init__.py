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
from .output_interpret import interpret_external_outputs, load_external_outputs
from .model_package import (
    ModelPackageArtifacts,
    build_model_package_id,
    export_model_package,
    extract_torch_export_bindings,
    validate_model_package_dir,
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
    "load_external_outputs",
    "interpret_external_outputs",
    "ModelPackageArtifacts",
    "build_model_package_id",
    "export_model_package",
    "extract_torch_export_bindings",
    "validate_model_package_dir",
]

__version__ = "0.2.0"
