"""Exact small-grid reference for fixed-threshold pixel repairs.

This Gate-C3a module intentionally does not build a max-tree and does not
provide a training loss.  It defines what a realizable binary repair costs
when every pixel whose threshold membership changes is charged.  Tree-based
critical actions must later reproduce this reference rather than redefine it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from skimage import measure


ActivationSemantics = Literal["infimum", "margin"]


@dataclass(frozen=True)
class PixelEditConfig:
    threshold_logit: float = 0.0
    activation_semantics: ActivationSemantics = "infimum"
    activation_margin: float = 1e-3
    margin_scale: float = 1.0
    max_pixels: int = 16
    connectivity: int = 2

    def validate(self) -> None:
        if not np.isfinite(self.threshold_logit):
            raise ValueError("threshold_logit must be finite")
        if self.activation_semantics not in {"infimum", "margin"}:
            raise ValueError("activation_semantics must be infimum or margin")
        if not np.isfinite(self.activation_margin) or self.activation_margin <= 0:
            raise ValueError("activation_margin must be finite and positive")
        if self.activation_semantics == "margin":
            with np.errstate(over="ignore", invalid="ignore"):
                activation_target = float(
                    self.threshold_logit + self.activation_margin
                )
            if (
                not np.isfinite(activation_target)
                or not activation_target > self.threshold_logit
            ):
                raise ValueError(
                    "threshold_logit + activation_margin must be finite "
                    "and strictly above the threshold"
                )
        if not np.isfinite(self.margin_scale) or self.margin_scale <= 0:
            raise ValueError("margin_scale must be finite and positive")
        if not isinstance(self.max_pixels, int) or isinstance(self.max_pixels, bool):
            raise ValueError("max_pixels must be a positive integer")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be a positive integer")
        if self.connectivity != 2:
            raise ValueError("connectivity must be 2 (8-connectivity for CCSR)")


@dataclass(frozen=True)
class PixelEditState:
    config: PixelEditConfig
    shape: tuple[int, int]
    source_logits: tuple[float, ...]
    mask_bits: int
    edit_energy: float
    activation_indices: tuple[int, ...]
    deactivation_indices: tuple[int, ...]
    infimum_attained: bool
    num_components: int

    @property
    def num_actions(self) -> int:
        return len(self.activation_indices) + len(self.deactivation_indices)

    def mask(self) -> np.ndarray:
        flat = np.fromiter(
            (
                bool((self.mask_bits >> index) & 1)
                for index in range(self.shape[0] * self.shape[1])
            ),
            dtype=bool,
            count=self.shape[0] * self.shape[1],
        )
        return flat.reshape(self.shape)


def _finite_logits_2d(logits) -> np.ndarray:
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    array = np.asarray(logits)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("logits must be a non-empty 2-D array")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError("logits must be numeric") from exc
    if not bool(np.all(finite)):
        raise ValueError("logits must contain only finite values")
    return array.astype(np.float64, copy=False)


def _binary_mask_2d(mask, *, shape: tuple[int, int]) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.shape != shape or array.ndim != 2:
        raise ValueError("desired_mask must be 2-D and match logits")
    if not bool(np.all((array == 0) | (array == 1))):
        raise ValueError("desired_mask must be binary")
    return array.astype(bool, copy=False)


def mask_to_bits(mask) -> int:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be 2-D")
    bits = 0
    for index, active in enumerate(array.reshape(-1)):
        if bool(active):
            bits |= 1 << index
    return bits


def build_pixel_edit_state(
    logits,
    desired_mask,
    *,
    config: PixelEditConfig | None = None,
) -> PixelEditState:
    """Return the exact charged membership edit for one desired mask."""
    config = PixelEditConfig() if config is None else config
    config.validate()
    logit_array = _finite_logits_2d(logits)
    if logit_array.size > config.max_pixels:
        raise ValueError(
            "pixel-edit reference supports at most %d pixels, got %d"
            % (config.max_pixels, logit_array.size)
        )
    desired = _binary_mask_2d(desired_mask, shape=logit_array.shape)
    raw = logit_array > config.threshold_logit
    activate = np.flatnonzero(desired & ~raw)
    deactivate = np.flatnonzero(~desired & raw)

    flat = logit_array.reshape(-1)
    if config.activation_semantics == "infimum":
        activation_cost = sum(
            config.threshold_logit - float(flat[index])
            for index in activate
        )
        infimum_attained = len(activate) == 0
    else:
        activation_cost = sum(
            config.threshold_logit
            + config.activation_margin
            - float(flat[index])
            for index in activate
        )
        infimum_attained = True
    deactivation_cost = sum(
        float(flat[index]) - config.threshold_logit
        for index in deactivate
    )
    energy = (activation_cost + deactivation_cost) / config.margin_scale
    if not np.isfinite(energy):
        raise ValueError("pixel edit energy must be finite")
    if energy < -1e-12:
        raise RuntimeError("pixel edit energy became negative")

    labels = measure.label(desired, connectivity=config.connectivity)
    return PixelEditState(
        config=config,
        shape=tuple(int(value) for value in desired.shape),
        source_logits=tuple(float(value) for value in flat),
        mask_bits=mask_to_bits(desired),
        edit_energy=float(max(0.0, energy)),
        activation_indices=tuple(int(index) for index in activate),
        deactivation_indices=tuple(int(index) for index in deactivate),
        infimum_attained=infimum_attained,
        num_components=int(labels.max()),
    )


def reconstruct_edited_logits(
    logits,
    state: PixelEditState,
    *,
    config: PixelEditConfig | None = None,
) -> np.ndarray:
    """Construct a finite edited logit image when the reference is attained."""
    config = state.config if config is None else config
    config.validate()
    if config != state.config:
        raise ValueError("reconstruction config must equal the state config")
    logit_array = _finite_logits_2d(logits).copy()
    if tuple(logit_array.shape) != state.shape:
        raise ValueError("state and logits shapes must match")
    if tuple(float(value) for value in logit_array.reshape(-1)) != state.source_logits:
        raise ValueError("state was built from different source logits")
    source_logits = logit_array.copy()
    if not state.infimum_attained:
        raise RuntimeError(
            "strict-threshold activation infimum is not attained; use margin semantics"
        )

    flat = logit_array.reshape(-1)
    for index in state.deactivation_indices:
        flat[index] = config.threshold_logit
    for index in state.activation_indices:
        flat[index] = config.threshold_logit + config.activation_margin
    reconstructed = flat.reshape(state.shape)
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("edited logits are not finite")
    if not np.array_equal(
        reconstructed > config.threshold_logit,
        state.mask(),
    ):
        raise RuntimeError("edited logits do not realize the requested mask")
    action_indices = set(state.activation_indices) | set(
        state.deactivation_indices
    )
    for index, (before, after) in enumerate(
        zip(source_logits.reshape(-1), reconstructed.reshape(-1))
    ):
        if index not in action_indices and before != after:
            raise RuntimeError("a pixel outside the action set changed")
    actual_energy = float(
        np.abs(reconstructed - source_logits).sum() / config.margin_scale
    )
    if not np.isclose(actual_energy, state.edit_energy, atol=1e-12, rtol=1e-12):
        raise RuntimeError(
            "reconstructed L1 energy does not match the state energy"
        )
    return reconstructed


def enumerate_pixel_edit_states(
    logits,
    *,
    config: PixelEditConfig | None = None,
) -> tuple[PixelEditState, ...]:
    """Exhaustively enumerate every binary repair on a tiny canvas."""
    config = PixelEditConfig() if config is None else config
    config.validate()
    logit_array = _finite_logits_2d(logits)
    if logit_array.size > config.max_pixels:
        raise ValueError(
            "pixel-edit reference supports at most %d pixels, got %d"
            % (config.max_pixels, logit_array.size)
        )

    states = []
    for bits in range(1 << logit_array.size):
        desired = np.fromiter(
            (
                bool((bits >> index) & 1)
                for index in range(logit_array.size)
            ),
            dtype=bool,
            count=logit_array.size,
        ).reshape(logit_array.shape)
        states.append(build_pixel_edit_state(logit_array, desired, config=config))
    return tuple(states)


__all__ = [
    "PixelEditConfig",
    "PixelEditState",
    "build_pixel_edit_state",
    "enumerate_pixel_edit_states",
    "mask_to_bits",
    "reconstruct_edited_logits",
]
