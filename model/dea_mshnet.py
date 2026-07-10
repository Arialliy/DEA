"""Canonical DEA main-model entry built on the open-source MSHNet encoder.

The implementation lives in ``predictive_correction_mshnet`` so checkpoints
from the initial mechanics run remain loadable.  ``DEAMSHNet`` is the public
model name; the older class name is a compatibility detail, not DEA-lite.
"""

from model.predictive_correction_mshnet import (
    PredictiveCorrectionMSHNet,
    TiedPredictionOperator,
)


class DEAMSHNet(PredictiveCorrectionMSHNet):
    """DEA v0: shared adjoint predictive-error correction decoder."""


__all__ = ["DEAMSHNet", "TiedPredictionOperator"]
