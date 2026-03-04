"""Source package for training and utilities.

This file makes the `src` folder importable as a package so tests
can use `from src.train import ...`.
"""

__all__ = ["train", "data_processing", "s3_utils", "evaluate"]
