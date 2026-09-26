"""
teacher_cache_v1.py
===================

Persistent cache for statistical-teacher supervision used by
`neural_relation_v3.py`.

Why this module exists
----------------------
The v3 statistical teacher uses horizon-matched rolling-origin validation.
Generating teacher targets can therefore be much more expensive than training
or re-training the Relation Neuron itself.  Teacher supervision should be
generated once and reused when the source data and teacher configuration are
unchanged.

This module stores `RelationTeacherDataset` in a portable compressed `.npz`
file without pickle.

Typical workflow
----------------
    from teacher_cache_v1 import (
        TeacherCacheConfig,
        get_or_build_teacher_cache,
    )

    config = TeacherCacheConfig(
        lookback=60,
        horizon=5,
        stride=5,
        teacher_kwargs={
            "validation_ratio": 0.20,
            "validation_stride": 1,
            "max_validation_origins": 10,
            "min_validation_success_rate": 0.8,
        },
    )

    cache = get_or_build_teacher_cache(
        Y,
        "teacher_cache/teacher_h5.npz",
        config=config,
    )

    teacher_dataset = cache.dataset

Cache validity
--------------
A cache is considered reusable only when BOTH are unchanged:

    1. source-data SHA256
    2. teacher-configuration SHA256

The configuration signature includes the statistical candidate keys and
fingerprints of optional custom relation factories.  This prevents silently
reusing, for example, an h=5 cache for h=10 or a cache built with a different
candidate library.

Stored supervision
------------------
For every TeacherSample:

    y_i, y_j
    i, j, origin, horizon
    family_probabilities
    expert_relations
    expert_valid
    relation_target
    weight_target
    null_probability
    selected_family
    selected_model
    teacher_validation_loss
    teacher_null_loss

The file also stores family_names and JSON metadata.

Dependency direction
--------------------
`teacher_cache_v1.py` may call `neural_relation_v3.py` lazily to construct or
reconstruct teacher datasets.  `neural_relation_v3.py` does NOT need to import
this module, avoiding a circular dependency.  Training/demo code can choose
whether to use caching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from pairwise_relation import (
    RelationFactory,
    get_relation_candidate_keys,
)


CACHE_FORMAT_NAME = "ese_teacher_cache"
CACHE_FORMAT_VERSION = 1


# ============================================================
# Configuration / results
# ============================================================


@dataclass(frozen=True)
class TeacherCacheConfig:
    """Configuration that uniquely determines statistical-teacher targets.

    Parameters mirror `neural_relation_v3.build_statistical_teacher_dataset`.

    `extra_relations` is intentionally excluded from normal dataclass
    serialization because Python callables are not portable JSON objects.
    Stable fingerprints of those factories are added to the cache signature
    separately.
    """

    lookback: int
    horizon: int = 1
    stride: int = 1

    start_origin: Optional[int] = None
    end_origin: Optional[int] = None

    teacher_temperature: float = 0.5
    epsilon: float = 1e-8

    max_pairs_per_origin: Optional[int] = None
    random_state: int = 42

    teacher_kwargs: Mapping[str, Any] = field(default_factory=dict)

    enable_extra: bool = False
    extra_relations: Optional[Sequence[RelationFactory]] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def validate(self) -> None:
        if self.lookback < 12:
            raise ValueError("lookback must be at least 12.")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if self.stride < 1:
            raise ValueError("stride must be at least 1.")
        if self.teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if (
            self.max_pairs_per_origin is not None
            and self.max_pairs_per_origin < 1
        ):
            raise ValueError("max_pairs_per_origin must be positive or None.")

    def build_kwargs(self) -> Dict[str, Any]:
        """Arguments for build_statistical_teacher_dataset(...)."""
        self.validate()

        return {
            "lookback": int(self.lookback),
            "horizon": int(self.horizon),
            "stride": int(self.stride),
            "start_origin": self.start_origin,
            "end_origin": self.end_origin,
            "teacher_temperature": float(self.teacher_temperature),
            "epsilon": float(self.epsilon),
            "max_pairs_per_origin": self.max_pairs_per_origin,
            "random_state": int(self.random_state),
            "teacher_kwargs": dict(self.teacher_kwargs),
            "enable_extra": bool(self.enable_extra),
            "extra_relations": self.extra_relations,
        }


@dataclass
class TeacherCacheMetadata:
    """Serializable metadata stored beside teacher arrays."""

    format_name: str
    format_version: int
    created_utc: str

    source_data_sha256: str
    source_shape: Tuple[int, int]
    source_dtype: str

    config_sha256: str
    cache_signature: str

    family_names: Sequence[str]
    n_samples: int
    lookback: int
    horizon: int

    candidate_keys: Sequence[str]
    extra_relation_fingerprints: Sequence[str]

    config: Mapping[str, Any]


@dataclass
class TeacherCacheResult:
    """Return value from `get_or_build_teacher_cache`."""

    dataset: Any
    metadata: TeacherCacheMetadata
    path: Path
    cache_hit: bool


# ============================================================
# Stable hashing / canonical JSON
# ============================================================


def _validate_Y(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y)

    if Y.ndim != 2:
        raise ValueError("Y must have shape (T, N).")
    if Y.shape[0] < 12:
        raise ValueError("Y must contain at least 12 observations.")
    if Y.shape[1] < 2:
        raise ValueError("Y must contain at least two systems.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values.")

    return Y


def hash_source_data(Y: np.ndarray) -> str:
    """Stable SHA256 of shape, dtype-independent float64 values, and ordering.

    Converting to little-endian float64 makes equivalent numerical arrays hash
    identically even if their original NumPy float dtype differs.
    """
    Y = _validate_Y(Y)

    canonical = np.ascontiguousarray(
        np.asarray(Y, dtype="<f8")
    )

    h = sha256()
    h.update(b"ESE_TEACHER_SOURCE_V1\0")
    h.update(
        np.asarray(canonical.shape, dtype="<i8").tobytes()
    )
    h.update(canonical.tobytes(order="C"))
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert common Python/NumPy values to deterministic JSON-safe objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(value[k])
            for k in sorted(value, key=lambda x: str(x))
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]

    if isinstance(value, set):
        return sorted(
            (_json_safe(x) for x in value),
            key=lambda x: json.dumps(
                x,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    # Do not silently discard an unusual teacher option.  Its stable-ish
    # textual representation still participates in the configuration hash.
    return {
        "__python_type__": (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}"
        ),
        "__repr__": repr(value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _factory_fingerprint(factory: RelationFactory) -> str:
    """Fingerprint a custom statistical relation factory.

    The fingerprint incorporates module/name information and, when available,
    its source code.  This is not intended as a security primitive; it is a
    cache-invalidation signature.
    """
    module = getattr(factory, "__module__", type(factory).__module__)
    qualname = getattr(
        factory,
        "__qualname__",
        getattr(factory, "__name__", type(factory).__qualname__),
    )

    try:
        source = inspect.getsource(factory)
    except (OSError, TypeError):
        source = repr(factory)

    payload = (
        f"{module}\n"
        f"{qualname}\n"
        f"{source}"
    ).encode("utf-8", errors="replace")

    digest = sha256(payload).hexdigest()
    return f"{module}.{qualname}:{digest}"


def _extra_relation_fingerprints(
    extra_relations: Optional[Sequence[RelationFactory]],
) -> Sequence[str]:
    if not extra_relations:
        return []
    return [_factory_fingerprint(x) for x in extra_relations]


def _config_payload(config: TeacherCacheConfig) -> Dict[str, Any]:
    config.validate()

    candidate_keys = get_relation_candidate_keys(
        enable_extra=config.enable_extra,
        extra_relations=config.extra_relations,
    )

    extra_fingerprints = _extra_relation_fingerprints(
        config.extra_relations
    )

    # Explicitly list fields rather than using asdict(), because asdict would
    # recursively touch callable factories.
    public_config = {
        "lookback": int(config.lookback),
        "horizon": int(config.horizon),
        "stride": int(config.stride),
        "start_origin": config.start_origin,
        "end_origin": config.end_origin,
        "teacher_temperature": float(config.teacher_temperature),
        "epsilon": float(config.epsilon),
        "max_pairs_per_origin": config.max_pairs_per_origin,
        "random_state": int(config.random_state),
        "teacher_kwargs": dict(config.teacher_kwargs),
        "enable_extra": bool(config.enable_extra),
        "candidate_keys": list(candidate_keys),
        "extra_relation_fingerprints": list(extra_fingerprints),
    }

    return _json_safe(public_config)


def hash_teacher_config(config: TeacherCacheConfig) -> str:
    """SHA256 of all teacher-generation choices that affect supervision."""
    payload = _canonical_json(
        _config_payload(config)
    ).encode("utf-8")

    h = sha256()
    h.update(b"ESE_TEACHER_CONFIG_V1\0")
    h.update(payload)
    return h.hexdigest()


def make_cache_signature(
    Y: np.ndarray,
    config: TeacherCacheConfig,
) -> str:
    """Combined source-data + configuration signature."""
    source_hash = hash_source_data(Y)
    config_hash = hash_teacher_config(config)

    h = sha256()
    h.update(b"ESE_TEACHER_CACHE_SIGNATURE_V1\0")
    h.update(source_hash.encode("ascii"))
    h.update(config_hash.encode("ascii"))
    return h.hexdigest()


def make_teacher_cache_path(
    cache_dir: str | Path,
    Y: np.ndarray,
    config: TeacherCacheConfig,
    *,
    prefix: str = "teacher",
    signature_chars: int = 16,
) -> Path:
    """Construct a deterministic cache filename.

    Example:
        teacher_h5_lb60_a1b2c3d4e5f6....npz
    """
    if signature_chars < 8:
        raise ValueError("signature_chars must be at least 8.")

    signature = make_cache_signature(Y, config)

    name = (
        f"{prefix}"
        f"_h{config.horizon}"
        f"_lb{config.lookback}"
        f"_{signature[:signature_chars]}"
        ".npz"
    )

    return Path(cache_dir) / name


# ============================================================
# Metadata
# ============================================================


def _make_metadata(
    Y: np.ndarray,
    config: TeacherCacheConfig,
    dataset: Any,
) -> TeacherCacheMetadata:
    source_hash = hash_source_data(Y)
    config_hash = hash_teacher_config(config)
    signature = make_cache_signature(Y, config)

    candidate_keys = get_relation_candidate_keys(
        enable_extra=config.enable_extra,
        extra_relations=config.extra_relations,
    )

    family_names = [str(x) for x in dataset.family_names]

    if family_names != list(candidate_keys):
        raise ValueError(
            "Dataset family_names do not match the candidate ordering implied "
            "by TeacherCacheConfig."
        )

    return TeacherCacheMetadata(
        format_name=CACHE_FORMAT_NAME,
        format_version=CACHE_FORMAT_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),

        source_data_sha256=source_hash,
        source_shape=(
            int(Y.shape[0]),
            int(Y.shape[1]),
        ),
        source_dtype=str(np.asarray(Y).dtype),

        config_sha256=config_hash,
        cache_signature=signature,

        family_names=family_names,
        n_samples=int(len(dataset)),
        lookback=int(dataset.lookback),
        horizon=int(config.horizon),

        candidate_keys=list(candidate_keys),
        extra_relation_fingerprints=list(
            _extra_relation_fingerprints(
                config.extra_relations
            )
        ),

        config=_config_payload(config),
    )


def _metadata_to_json(metadata: TeacherCacheMetadata) -> str:
    return _canonical_json(asdict(metadata))


def _metadata_from_json(text: str) -> TeacherCacheMetadata:
    raw = json.loads(str(text))

    required = {
        "format_name",
        "format_version",
        "created_utc",
        "source_data_sha256",
        "source_shape",
        "source_dtype",
        "config_sha256",
        "cache_signature",
        "family_names",
        "n_samples",
        "lookback",
        "horizon",
        "candidate_keys",
        "extra_relation_fingerprints",
        "config",
    }

    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "Teacher cache metadata is missing fields: "
            + ", ".join(missing)
        )

    if raw["format_name"] != CACHE_FORMAT_NAME:
        raise ValueError(
            f"Unsupported teacher cache format: {raw['format_name']!r}."
        )

    if int(raw["format_version"]) != CACHE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported teacher cache version: "
            f"{raw['format_version']}; expected {CACHE_FORMAT_VERSION}."
        )

    shape = tuple(int(x) for x in raw["source_shape"])
    if len(shape) != 2:
        raise ValueError("source_shape in teacher cache must have length 2.")

    return TeacherCacheMetadata(
        format_name=str(raw["format_name"]),
        format_version=int(raw["format_version"]),
        created_utc=str(raw["created_utc"]),

        source_data_sha256=str(raw["source_data_sha256"]),
        source_shape=(shape[0], shape[1]),
        source_dtype=str(raw["source_dtype"]),

        config_sha256=str(raw["config_sha256"]),
        cache_signature=str(raw["cache_signature"]),

        family_names=[str(x) for x in raw["family_names"]],
        n_samples=int(raw["n_samples"]),
        lookback=int(raw["lookback"]),
        horizon=int(raw["horizon"]),

        candidate_keys=[str(x) for x in raw["candidate_keys"]],
        extra_relation_fingerprints=[
            str(x) for x in raw["extra_relation_fingerprints"]
        ],

        config=dict(raw["config"]),
    )


# ============================================================
# Dataset <-> arrays
# ============================================================


def _dataset_to_arrays(dataset: Any) -> Dict[str, np.ndarray]:
    samples = list(dataset.samples)

    if not samples:
        raise ValueError("Cannot cache an empty teacher dataset.")

    return {
        "y_i": np.stack(
            [np.asarray(s.y_i, dtype=np.float32) for s in samples]
        ),
        "y_j": np.stack(
            [np.asarray(s.y_j, dtype=np.float32) for s in samples]
        ),

        "i": np.asarray([s.i for s in samples], dtype=np.int64),
        "j": np.asarray([s.j for s in samples], dtype=np.int64),
        "origin": np.asarray(
            [s.origin for s in samples],
            dtype=np.int64,
        ),
        "horizon": np.asarray(
            [s.horizon for s in samples],
            dtype=np.int64,
        ),

        "family_probabilities": np.stack(
            [
                np.asarray(
                    s.family_probabilities,
                    dtype=np.float32,
                )
                for s in samples
            ]
        ),
        "expert_relations": np.stack(
            [
                np.asarray(
                    s.expert_relations,
                    dtype=np.float32,
                )
                for s in samples
            ]
        ),
        "expert_valid": np.stack(
            [
                np.asarray(
                    s.expert_valid,
                    dtype=bool,
                )
                for s in samples
            ]
        ),

        "relation_target": np.asarray(
            [s.relation_target for s in samples],
            dtype=np.float64,
        ),
        "weight_target": np.asarray(
            [s.weight_target for s in samples],
            dtype=np.float64,
        ),

        "null_probability": np.asarray(
            [s.null_probability for s in samples],
            dtype=np.float64,
        ),

        "selected_family": np.asarray(
            [str(s.selected_family) for s in samples],
            dtype=np.str_,
        ),
        "selected_model": np.asarray(
            [str(s.selected_model) for s in samples],
            dtype=np.str_,
        ),

        "teacher_validation_loss": np.asarray(
            [s.teacher_validation_loss for s in samples],
            dtype=np.float64,
        ),
        "teacher_null_loss": np.asarray(
            [s.teacher_null_loss for s in samples],
            dtype=np.float64,
        ),

        "family_names": np.asarray(
            [str(x) for x in dataset.family_names],
            dtype=np.str_,
        ),
        "epsilon": np.asarray(
            float(dataset.epsilon),
            dtype=np.float64,
        ),
    }


def _validate_cached_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: TeacherCacheMetadata,
) -> None:
    required = {
        "y_i",
        "y_j",
        "i",
        "j",
        "origin",
        "horizon",
        "family_probabilities",
        "expert_relations",
        "expert_valid",
        "relation_target",
        "weight_target",
        "null_probability",
        "selected_family",
        "selected_model",
        "teacher_validation_loss",
        "teacher_null_loss",
        "family_names",
        "epsilon",
    }

    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(
            "Teacher cache is missing arrays: "
            + ", ".join(missing)
        )

    n = int(metadata.n_samples)
    k = len(metadata.family_names)
    lookback = int(metadata.lookback)

    if arrays["y_i"].shape != (n, lookback):
        raise ValueError(
            "Cached y_i has invalid shape: "
            f"{arrays['y_i'].shape}; expected {(n, lookback)}."
        )
    if arrays["y_j"].shape != (n, lookback):
        raise ValueError(
            "Cached y_j has invalid shape: "
            f"{arrays['y_j'].shape}; expected {(n, lookback)}."
        )

    for name in (
        "i",
        "j",
        "origin",
        "horizon",
        "relation_target",
        "weight_target",
        "null_probability",
        "selected_family",
        "selected_model",
        "teacher_validation_loss",
        "teacher_null_loss",
    ):
        if arrays[name].shape != (n,):
            raise ValueError(
                f"Cached {name} has shape {arrays[name].shape}; "
                f"expected {(n,)}."
            )

    if arrays["family_probabilities"].shape != (n, k + 1):
        raise ValueError(
            "Cached family_probabilities has invalid shape."
        )
    if arrays["expert_relations"].shape != (n, k):
        raise ValueError(
            "Cached expert_relations has invalid shape."
        )
    if arrays["expert_valid"].shape != (n, k):
        raise ValueError(
            "Cached expert_valid has invalid shape."
        )

    cached_family_names = [
        str(x) for x in arrays["family_names"].tolist()
    ]
    if cached_family_names != list(metadata.family_names):
        raise ValueError(
            "Cached family_names disagree with metadata."
        )

    if not np.all(
        np.asarray(arrays["horizon"], dtype=np.int64)
        == int(metadata.horizon)
    ):
        raise ValueError(
            "Teacher cache contains samples with unexpected horizons."
        )

    probabilities = np.asarray(
        arrays["family_probabilities"],
        dtype=float,
    )
    probability_sums = probabilities.sum(axis=1)

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -1e-7)
        or not np.allclose(
            probability_sums,
            1.0,
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise ValueError(
            "Cached family_probabilities are invalid."
        )

    weights = np.asarray(arrays["weight_target"], dtype=float)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < -1e-8)
        or np.any(weights > 1.0 + 1e-8)
    ):
        raise ValueError(
            "Cached weight_target values must lie in [0,1]."
        )


def _arrays_to_dataset(
    arrays: Mapping[str, np.ndarray],
    metadata: TeacherCacheMetadata,
) -> Any:
    # Lazy import is deliberate.  It avoids requiring neural_relation_v3 to
    # import this cache module and therefore avoids a circular dependency.
    from neural_relation import (
        RelationTeacherDataset,
        TeacherSample,
    )

    _validate_cached_arrays(arrays, metadata)

    n = int(metadata.n_samples)

    samples = []

    for idx in range(n):
        samples.append(
            TeacherSample(
                y_i=np.asarray(
                    arrays["y_i"][idx],
                    dtype=np.float32,
                ).copy(),
                y_j=np.asarray(
                    arrays["y_j"][idx],
                    dtype=np.float32,
                ).copy(),

                i=int(arrays["i"][idx]),
                j=int(arrays["j"][idx]),
                origin=int(arrays["origin"][idx]),
                horizon=int(arrays["horizon"][idx]),

                family_probabilities=np.asarray(
                    arrays["family_probabilities"][idx],
                    dtype=np.float32,
                ).copy(),
                expert_relations=np.asarray(
                    arrays["expert_relations"][idx],
                    dtype=np.float32,
                ).copy(),
                expert_valid=np.asarray(
                    arrays["expert_valid"][idx],
                    dtype=bool,
                ).copy(),

                relation_target=float(
                    arrays["relation_target"][idx]
                ),
                weight_target=float(
                    arrays["weight_target"][idx]
                ),

                null_probability=float(
                    arrays["null_probability"][idx]
                ),
                selected_family=str(
                    arrays["selected_family"][idx]
                ),
                selected_model=str(
                    arrays["selected_model"][idx]
                ),
                teacher_validation_loss=float(
                    arrays["teacher_validation_loss"][idx]
                ),
                teacher_null_loss=float(
                    arrays["teacher_null_loss"][idx]
                ),
            )
        )

    return RelationTeacherDataset(
        samples=samples,
        family_names=list(metadata.family_names),
        epsilon=float(np.asarray(arrays["epsilon"]).item()),
    )


# ============================================================
# Save / load
# ============================================================


def save_teacher_cache(
    dataset: Any,
    path: str | Path,
    *,
    Y: np.ndarray,
    config: TeacherCacheConfig,
    overwrite: bool = False,
) -> TeacherCacheMetadata:
    """Persist a statistical-teacher dataset as compressed NPZ."""
    Y = _validate_Y(Y)
    config.validate()

    path = Path(path)

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Teacher cache already exists: {path}. "
            "Use overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _make_metadata(
        Y,
        config,
        dataset,
    )
    arrays = _dataset_to_arrays(dataset)

    # Revalidate before writing so corrupted/inconsistent objects are rejected.
    _validate_cached_arrays(
        arrays,
        metadata,
    )

    np.savez_compressed(
        path,
        metadata_json=np.asarray(
            _metadata_to_json(metadata),
            dtype=np.str_,
        ),
        **arrays,
    )

    return metadata


def read_teacher_cache_metadata(
    path: str | Path,
) -> TeacherCacheMetadata:
    """Read only metadata; teacher samples are not reconstructed."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Teacher cache does not exist: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(
                "File is not a valid ESE teacher cache: "
                "metadata_json is missing."
            )

        metadata_text = str(
            np.asarray(data["metadata_json"]).item()
        )

    return _metadata_from_json(metadata_text)


def teacher_cache_matches(
    path: str | Path,
    *,
    Y: np.ndarray,
    config: TeacherCacheConfig,
) -> bool:
    """Return True only when source data and teacher configuration both match."""
    try:
        metadata = read_teacher_cache_metadata(path)
    except (FileNotFoundError, ValueError, OSError):
        return False

    try:
        source_hash = hash_source_data(Y)
        config_hash = hash_teacher_config(config)
    except Exception:
        return False

    return bool(
        metadata.source_data_sha256 == source_hash
        and metadata.config_sha256 == config_hash
        and metadata.cache_signature
        == make_cache_signature(Y, config)
    )


def _validate_expected_cache(
    metadata: TeacherCacheMetadata,
    *,
    expected_Y: Optional[np.ndarray],
    expected_config: Optional[TeacherCacheConfig],
) -> None:
    if expected_Y is not None:
        expected_source_hash = hash_source_data(
            expected_Y
        )
        if metadata.source_data_sha256 != expected_source_hash:
            raise ValueError(
                "Teacher cache source-data hash does not match the "
                "provided dataset."
            )

    if expected_config is not None:
        expected_config_hash = hash_teacher_config(
            expected_config
        )
        if metadata.config_sha256 != expected_config_hash:
            raise ValueError(
                "Teacher cache configuration hash does not match the "
                "requested teacher configuration."
            )

        expected_signature = (
            make_cache_signature(
                expected_Y,
                expected_config,
            )
            if expected_Y is not None
            else None
        )

        if (
            expected_signature is not None
            and metadata.cache_signature
            != expected_signature
        ):
            raise ValueError(
                "Teacher cache signature does not match the requested "
                "source/configuration pair."
            )


def load_teacher_cache(
    path: str | Path,
    *,
    expected_Y: Optional[np.ndarray] = None,
    expected_config: Optional[TeacherCacheConfig] = None,
) -> Tuple[Any, TeacherCacheMetadata]:
    """Load and reconstruct RelationTeacherDataset.

    Pass `expected_Y` and `expected_config` in formal experiments so stale
    caches are rejected rather than silently reused.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Teacher cache does not exist: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(
                "File is not a valid ESE teacher cache: "
                "metadata_json is missing."
            )

        metadata = _metadata_from_json(
            str(
                np.asarray(
                    data["metadata_json"]
                ).item()
            )
        )

        _validate_expected_cache(
            metadata,
            expected_Y=expected_Y,
            expected_config=expected_config,
        )

        arrays = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "metadata_json"
        }

    dataset = _arrays_to_dataset(
        arrays,
        metadata,
    )

    return dataset, metadata


# ============================================================
# Build-or-load workflow
# ============================================================


def get_or_build_teacher_cache(
    Y: np.ndarray,
    path: str | Path,
    *,
    config: TeacherCacheConfig,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> TeacherCacheResult:
    """Load a valid teacher cache or generate/save it once.

    Cache-hit rule
    --------------
    A file is reused only when its source-data and teacher-configuration hashes
    match the current request.

    If a file exists but is stale, it is rebuilt in place.  The stale file is
    never treated as valid supervision.
    """
    Y = _validate_Y(Y)
    config.validate()

    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    if (
        path.exists()
        and not force_rebuild
        and teacher_cache_matches(
            path,
            Y=Y,
            config=config,
        )
    ):
        dataset, metadata = load_teacher_cache(
            path,
            expected_Y=Y,
            expected_config=config,
        )

        if verbose:
            print(
                "[teacher-cache] HIT "
                f"{path} | samples={len(dataset)} "
                f"| h={metadata.horizon} "
                f"| lookback={metadata.lookback}"
            )

        return TeacherCacheResult(
            dataset=dataset,
            metadata=metadata,
            path=path,
            cache_hit=True,
        )

    if verbose:
        reason = (
            "forced rebuild"
            if force_rebuild
            else (
                "cache missing"
                if not path.exists()
                else "cache stale / configuration changed"
            )
        )
        print(
            f"[teacher-cache] MISS ({reason}) -> "
            "building statistical teacher..."
        )

    # Lazy import preserves a one-way optional dependency.
    from neural_relation import (
        build_statistical_teacher_dataset,
    )

    dataset = build_statistical_teacher_dataset(
        Y,
        verbose=verbose,
        **config.build_kwargs(),
    )

    metadata = save_teacher_cache(
        dataset,
        path,
        Y=Y,
        config=config,
        overwrite=True,
    )

    if verbose:
        print(
            "[teacher-cache] SAVED "
            f"{path} | samples={len(dataset)} "
            f"| signature={metadata.cache_signature[:16]}"
        )

    return TeacherCacheResult(
        dataset=dataset,
        metadata=metadata,
        path=path,
        cache_hit=False,
    )


# ============================================================
# Reporting
# ============================================================


def teacher_cache_summary(
    metadata: TeacherCacheMetadata,
) -> Dict[str, Any]:
    """Compact metadata for logging or experiment tables."""
    return {
        "format_version": int(metadata.format_version),
        "created_utc": metadata.created_utc,
        "source_shape": tuple(metadata.source_shape),
        "n_samples": int(metadata.n_samples),
        "lookback": int(metadata.lookback),
        "horizon": int(metadata.horizon),
        "n_families": len(metadata.family_names),
        "family_names": list(metadata.family_names),
        "source_data_sha256": metadata.source_data_sha256,
        "config_sha256": metadata.config_sha256,
        "cache_signature": metadata.cache_signature,
    }


def print_teacher_cache_summary(
    metadata: TeacherCacheMetadata,
    *,
    path: Optional[str | Path] = None,
) -> None:
    """Human-readable cache metadata."""
    print("=" * 76)
    print("Statistical Teacher Cache")
    print("=" * 76)

    if path is not None:
        print(f"Path          : {Path(path)}")

    print(f"Format        : v{metadata.format_version}")
    print(f"Created UTC   : {metadata.created_utc}")
    print(f"Source shape  : {metadata.source_shape}")
    print(f"Samples       : {metadata.n_samples}")
    print(f"Lookback      : {metadata.lookback}")
    print(f"Horizon       : {metadata.horizon}")
    print(
        "Families      : "
        + ", ".join(metadata.family_names)
    )
    print(
        f"Data hash     : "
        f"{metadata.source_data_sha256[:16]}..."
    )
    print(
        f"Config hash   : "
        f"{metadata.config_sha256[:16]}..."
    )
    print(
        f"Signature     : "
        f"{metadata.cache_signature[:16]}..."
    )
    print("=" * 76)