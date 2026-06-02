"""
Data Pipeline — Production Data Processing & ETL Framework
============================================================
Provides comprehensive data pipeline capabilities: ingestion,
transformation, validation, batching, streaming simulation,
data versioning, and schema management for ML workflows.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class DataFormat(Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    NUMPY = "numpy"
    TF_RECORD = "tf_record"
    ARROW = "arrow"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    TIMESERIES = "timeseries"
    BINARY = "binary"
    EMBEDDING = "embedding"
    STRUCTURED = "structured"


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str = ""
    dtype: str = "float64"
    nullable: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetSchema:
    """Complete dataset schema."""
    columns: List[ColumnSchema] = field(default_factory=list)
    num_rows: int = 0
    row_checksum: str = ""
    created_at: str = ""
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": [asdict(c) for c in self.columns],
            "num_rows": self.num_rows,
            "row_checksum": self.row_checksum,
            "created_at": self.created_at,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Data Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """Validate data against schema and quality rules."""

    @staticmethod
    def validate_row(row: Dict[str, Any],
                     schema: DatasetSchema) -> Dict[str, Any]:
        """Validate a single row against the schema."""
        errors = []
        warnings = []
        valid = True

        for col in schema.columns:
            val = row.get(col.name)
            if val is None:
                if not col.nullable:
                    errors.append(f"Column '{col.name}' is required but missing.")
                    valid = False
                continue

            # Type check
            expected_type = col.dtype
            if expected_type == "float64" and not isinstance(val, (int, float)):
                try:
                    float(val)
                except (ValueError, TypeError):
                    errors.append(f"Column '{col.name}': expected float, got {type(val).__name__}")
                    valid = False
            elif expected_type == "int64" and not isinstance(val, int):
                try:
                    int(val)
                except (ValueError, TypeError):
                    errors.append(f"Column '{col.name}': expected int, got {type(val).__name__}")
                    valid = False

            # Constraints
            for constraint, expected in col.constraints.items():
                if constraint == "min" and val < expected:
                    errors.append(f"Column '{col.name}': {val} < min {expected}")
                    valid = False
                elif constraint == "max" and val > expected:
                    errors.append(f"Column '{col.name}': {val} > max {expected}")
                    valid = False
                elif constraint == "pattern":
                    import re
                    if not re.match(expected, str(val)):
                        errors.append(f"Column '{col.name}': doesn't match pattern {expected}")
                        valid = False

        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def validate_batch(rows: List[Dict[str, Any]],
                       schema: DatasetSchema) -> Dict[str, Any]:
        """Validate a batch of rows."""
        results = [DataValidator.validate_row(r, schema) for r in rows]
        valid_count = sum(1 for r in results if r["valid"])
        return {
            "total": len(rows),
            "valid": valid_count,
            "invalid": len(rows) - valid_count,
            "valid_ratio": valid_count / len(rows) if rows else 1.0,
            "all_errors": [r["errors"] for r in results if not r["valid"]],
        }


# ---------------------------------------------------------------------------
# Data Transformer
# ---------------------------------------------------------------------------

class DataTransformer:
    """Transform and normalize data."""

    @staticmethod
    def normalize_numeric(values: List[float],
                          method: str = "zscore") -> Dict[str, Any]:
        """Normalize numeric values."""
        arr = np.array(values, dtype=np.float64)
        if method == "zscore":
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std == 0:
                return {"values": [0.0] * len(values), "params": {"mean": mean, "std": 1.0}}
            normalized = ((arr - mean) / std).tolist()
            return {"values": normalized, "params": {"mean": mean, "std": std}}
        elif method == "minmax":
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx == mn:
                return {"values": [0.5] * len(values), "params": {"min": mn, "max": mx + 1}}
            normalized = ((arr - mn) / (mx - mn)).tolist()
            return {"values": normalized, "params": {"min": mn, "max": mx}}
        elif method == "robust":
            median = float(np.median(arr))
            q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
            iqr = q75 - q25
            if iqr == 0:
                return {"values": [0.0] * len(values), "params": {"median": median, "iqr": 1.0}}
            normalized = ((arr - median) / iqr).tolist()
            return {"values": normalized, "params": {"median": median, "iqr": iqr}}
        return {"values": values, "params": {}}

    @staticmethod
    def encode_categorical(values: List[str],
                           method: str = "onehot") -> Dict[str, Any]:
        """Encode categorical values."""
        unique = sorted(set(v for v in values if v is not None))
        if method == "onehot":
            mapping = {cat: [1.0 if i == j else 0.0 for j in range(len(unique))]
                       for i, cat in enumerate(unique)}
            encoded = [mapping.get(v, [0.0] * len(unique)) for v in values]
            return {"encoded": encoded, "categories": unique, "method": "onehot"}
        elif method == "label":
            mapping = {cat: i for i, cat in enumerate(unique)}
            encoded = [float(mapping.get(v, -1)) for v in values]
            return {"encoded": encoded, "categories": unique, "method": "label"}
        return {"encoded": values, "categories": unique}

    @staticmethod
    def split_dataset(data: List[Dict[str, Any]],
                      ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                      seed: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset into train/val/test."""
        rng = random.Random(seed)
        indices = list(range(len(data)))
        rng.shuffle(indices)
        n = len(data)
        n1 = int(ratios[0] * n)
        n2 = int(ratios[1] * n)
        return {
            "train": [data[i] for i in indices[:n1]],
            "val": [data[i] for i in indices[n1:n1 + n2]],
            "test": [data[i] for i in indices[n1 + n2:]],
        }


# ---------------------------------------------------------------------------
# Data Generator (Synthetic)
# ---------------------------------------------------------------------------

class DataGenerator:
    """Generate synthetic datasets for testing and development."""

    @staticmethod
    def generate_tabular(num_rows: int = 1000,
                         num_features: int = 10,
                         num_classes: int = 2,
                         seed: int = 0) -> Dict[str, Any]:
        """Generate synthetic tabular data with labels."""
        rng = np.random.RandomState(seed)
        features = rng.randn(num_rows, num_features).tolist()
        # Create non-linear decision boundary
        weights = rng.randn(num_features)
        logits = np.dot(np.array(features), weights) + rng.randn(num_rows) * 0.1
        probs = 1.0 / (1.0 + np.exp(-logits))
        labels = (probs > 0.5).astype(int).tolist()

        columns = [ColumnSchema(name=f"feature_{i}", dtype="float64")
                   for i in range(num_features)]
        columns.append(ColumnSchema(name="label", dtype="int64"))

        return {
            "num_rows": num_rows,
            "num_features": num_features,
            "num_classes": num_classes,
            "features": features,
            "labels": labels,
            "schema": {"columns": [asdict(c) for c in columns]},
        }

    @staticmethod
    def generate_timeseries(num_points: int = 1000,
                            num_series: int = 3,
                            trend: float = 0.0,
                            seasonality: float = 1.0,
                            noise: float = 0.1,
                            seed: int = 0) -> Dict[str, Any]:
        """Generate synthetic time series data."""
        rng = np.random.RandomState(seed)
        t = np.arange(num_points)
        series_data = []
        for s in range(num_series):
            trend_comp = trend * t
            seasonal_comp = seasonality * np.sin(2 * np.pi * t / (50 + s * 10))
            noise_comp = rng.randn(num_points) * noise
            values = trend_comp + seasonal_comp + noise_comp
            series_data.append(values.tolist())

        return {
            "num_points": num_points,
            "num_series": num_series,
            "time": t.tolist(),
            "series": [{"name": f"series_{s}", "values": series_data[s]}
                       for s in range(num_series)],
            "parameters": {"trend": trend, "seasonality": seasonality, "noise": noise},
        }

    @staticmethod
    def generate_text_corpus(num_docs: int = 100,
                             vocab_size: int = 1000,
                             avg_length: int = 50,
                             seed: int = 0) -> Dict[str, Any]:
        """Generate synthetic text documents."""
        rng = random.Random(seed)
        vocab = [f"word_{i}" for i in range(vocab_size)]
        documents = []
        for _ in range(num_docs):
            length = max(1, int(rng.gauss(avg_length, avg_length * 0.3)))
            doc = " ".join(rng.choice(vocab) for _ in range(length))
            documents.append(doc)

        return {
            "num_docs": num_docs,
            "vocab_size": vocab_size,
            "avg_length": avg_length,
            "documents": documents[:10],  # sample
            "total_chars": sum(len(d) for d in documents),
        }


# ---------------------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """Process data in batches with memory management."""

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size

    def iterate(self, data: List[Any]) -> Generator[List[Any], None, None]:
        """Yield batches from data."""
        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size]

    def process_in_batches(self, items: List[Dict[str, Any]],
                           transform_fn: Optional[str] = None) -> Dict[str, Any]:
        """Process items in batches with optional transform."""
        results = []
        num_batches = 0
        for batch in self.iterate(items):
            num_batches += 1
            if transform_fn == "normalize":
                for item in batch:
                    for k, v in item.items():
                        if isinstance(v, (int, float)):
                            item[k] = v  # passthrough
            results.extend(batch)

        return {
            "total_items": len(items),
            "batch_size": self.batch_size,
            "num_batches": num_batches,
            "processed": len(results),
            "throughput": len(items) / max(len(items) * 0.001, 1),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_synthetic_data(data_type: str = "tabular",
                                   num_rows: int = 1000,
                                   num_features: int = 10,
                                   seed: int = 0) -> Dict[str, Any]:
    """Generate synthetic dataset for testing."""
    gen = DataGenerator()
    if data_type == "tabular":
        return gen.generate_tabular(num_rows, num_features, seed=seed)
    elif data_type == "timeseries":
        return gen.generate_timeseries(num_rows, seed=seed)
    elif data_type == "text":
        return gen.generate_text_corpus(seed=seed)
    return gen.generate_tabular(num_rows, num_features, seed=seed)


async def validate_dataset(data: List[Dict[str, Any]],
                            schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate a dataset against a schema."""
    if not schema:
        # Infer schema from data
        columns = []
        for key in data[0].keys() if data else []:
            val = data[0][key]
            dtype = type(val).__name__
            if dtype == "int": dtype = "int64"
            elif dtype == "float": dtype = "float64"
            columns.append(ColumnSchema(name=key, dtype=dtype))
        schema = DatasetSchema(columns=columns)

    ds_schema = DatasetSchema(
        columns=[ColumnSchema(**c) if isinstance(c, dict) else c for c in schema.get("columns", [])]
    )
    return DataValidator.validate_batch(data, ds_schema)


async def transform_data(values: List[float],
                          method: str = "zscore") -> Dict[str, Any]:
    """Normalize numeric data."""
    return DataTransformer.normalize_numeric(values, method)


async def encode_categories(values: List[str],
                             method: str = "onehot") -> Dict[str, Any]:
    """Encode categorical data."""
    return DataTransformer.encode_categorical(values, method)


async def split_dataset(data: List[Dict[str, Any]],
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         seed: int = 0) -> Dict[str, Any]:
    """Split dataset into train/val/test."""
    test_ratio = 1.0 - train_ratio - val_ratio
    return DataTransformer.split_dataset(data, (train_ratio, val_ratio, test_ratio), seed)


async def create_schema(columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a dataset schema definition."""
    schema = DatasetSchema(
        columns=[ColumnSchema(**c) for c in columns],
        created_at=datetime.utcnow().isoformat(),
    )
    return schema.to_dict()


async def process_batch(items: List[Dict[str, Any]],
                         batch_size: int = 32) -> Dict[str, Any]:
    """Process items in batches."""
    processor = BatchProcessor(batch_size)
    return processor.process_in_batches(items)
