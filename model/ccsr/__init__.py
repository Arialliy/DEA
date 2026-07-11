"""Reference building blocks for Component-Cut Structured Risk audits.

Only the discrete task risk is exposed at Gate C2.  Max-tree, structured
hinge, and training routes must not be added until their formal gates pass.
"""

from .task_risk import (
    ComponentEditRiskResult,
    ComponentMatch,
    build_components_from_binary,
    build_gt_components,
    component_pair_cost,
    exact_component_edit_risk,
    exact_component_edit_risk_from_masks,
)
from .pixel_edit_reference import (
    PixelEditConfig,
    PixelEditState,
    build_pixel_edit_state,
    enumerate_pixel_edit_states,
    reconstruct_edited_logits,
)
from .reference_solver import (
    StructuredCandidate,
    StructuredHingeResult,
    solve_exhaustive_structured_hinge,
)

__all__ = [
    "ComponentEditRiskResult",
    "ComponentMatch",
    "build_components_from_binary",
    "build_gt_components",
    "component_pair_cost",
    "exact_component_edit_risk",
    "exact_component_edit_risk_from_masks",
    "PixelEditConfig",
    "PixelEditState",
    "build_pixel_edit_state",
    "enumerate_pixel_edit_states",
    "reconstruct_edited_logits",
    "StructuredCandidate",
    "StructuredHingeResult",
    "solve_exhaustive_structured_hinge",
]
