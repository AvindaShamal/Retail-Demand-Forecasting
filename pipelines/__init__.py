"""Pipeline package exports.

Keep package initialization lightweight so importing a single pipeline module
does not eagerly import unrelated training dependencies during API startup.
"""

from pipelines.batch_inference_pipeline import get_logger
from pipelines.batch_inference_pipeline import load_model_from_registry
from pipelines.batch_inference_pipeline import run_batch_inference

__all__ = ["get_logger", "load_model_from_registry", "run_batch_inference"]
