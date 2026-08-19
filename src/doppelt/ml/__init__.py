"""State encoding, datasets, models, and training."""

from doppelt.ml.encoding import (
    ENCODING_VERSION,
    FEATURE_SIZE,
    SCORE_SCALE,
    EncodedState,
    encode_features,
    encode_state,
    legal_mask,
)

__all__ = [
    "ENCODING_VERSION",
    "FEATURE_SIZE",
    "SCORE_SCALE",
    "EncodedState",
    "encode_features",
    "encode_state",
    "legal_mask",
]
