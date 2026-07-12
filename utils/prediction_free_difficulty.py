"""Prediction-free image-and-annotation covariates for Gate E-1b.

The module deliberately accepts only RGB pixels and ground-truth masks.  It
does not accept model scores, features, checkpoints, or seed-specific
predictions.  Outcome rows are joined later by the canonical stable target
identifier.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy import optimize, stats
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from skimage import measure

from utils.target_identity import build_stable_target_set


COVARIATES = (
    "log1p_area",
    "border_distance",
    "local_robust_scr",
    "local_ring_robust_dispersion",
)
RESPONSES = ("miss_any_seed", "miss_three_of_three")
MIN_RING_PIXELS = 16
MAX_UNAVAILABLE_FRACTION = 0.05
LODO_L2_C = 1.0
MIN_CLASS_IMAGE_CLUSTERS = 10


class DifficultyAuditError(RuntimeError):
    """Raised when the frozen E-1b contract cannot be satisfied."""


def _binary_mask(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise DifficultyAuditError("target mask must be a non-empty 2-D array")
    if not bool(np.all((array == 0) | (array == 1))):
        raise DifficultyAuditError("target mask must be exactly binary")
    return array.astype(bool, copy=False)


def _rgb8(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise DifficultyAuditError("validation image must be HxWx3 RGB")
    if array.dtype != np.uint8:
        raise DifficultyAuditError("validation RGB image must be uint8 after conversion")
    return array


def compute_prediction_free_covariates(
    rgb8: object,
    target_mask: object,
    *,
    dataset: str,
    image_name: str,
) -> list[dict[str, Any]]:
    """Compute the four prospectively frozen covariates for every GT target."""

    rgb = _rgb8(rgb8)
    target = _binary_mask(target_mask)
    if rgb.shape[:2] != target.shape:
        raise DifficultyAuditError("RGB image and target mask shapes differ")
    intensity = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    ) / 255.0
    if not bool(np.isfinite(intensity).all()):
        raise DifficultyAuditError("derived intensity is non-finite")

    target_set = build_stable_target_set(
        target,
        dataset=dataset,
        image_name=image_name,
        connectivity=2,
    )
    labels = measure.label(target, connectivity=2)
    rows: list[dict[str, Any]] = []
    for identity in target_set.targets:
        component = labels == identity.source_label
        if int(component.sum()) != identity.area:
            raise DifficultyAuditError("component geometry disagrees with identity")
        coords = np.argwhere(component)
        border_distance = int(
            np.min(
                np.stack(
                    (
                        coords[:, 0],
                        coords[:, 1],
                        target.shape[0] - 1 - coords[:, 0],
                        target.shape[1] - 1 - coords[:, 1],
                    ),
                    axis=1,
                )
            )
        )
        distance = distance_transform_edt(~component)
        ring = (distance >= 2.0) & (distance < 9.0) & (~target)
        ring_count = int(ring.sum())
        if ring_count < MIN_RING_PIXELS:
            ring_median = None
            ring_dispersion = None
            scr = None
        else:
            ring_values = intensity[ring]
            ring_median_value = float(np.median(ring_values))
            dispersion_value = float(
                1.4826 * np.median(np.abs(ring_values - ring_median_value))
            )
            scr_value = float(
                (float(np.mean(intensity[component])) - ring_median_value)
                / (dispersion_value + 1e-6)
            )
            if not all(
                np.isfinite(value)
                for value in (ring_median_value, dispersion_value, scr_value)
            ):
                raise DifficultyAuditError("local robust covariate is non-finite")
            ring_median = ring_median_value
            ring_dispersion = dispersion_value
            scr = scr_value
        rows.append(
            {
                "schema_version": "dea.gate_e.prediction_free_covariates.v1",
                "dataset": dataset,
                "image_name": image_name,
                "stable_target_id": identity.stable_key,
                "component_mask_sha256": identity.component_mask_sha256,
                "label_mask_sha256": identity.label_mask_sha256,
                "component_index": identity.component_index,
                "source_component_index": identity.source_component_index,
                "source_label": identity.source_label,
                "area": identity.area,
                "ring_pixel_count": ring_count,
                "ring_available": ring_count >= MIN_RING_PIXELS,
                "ring_median_intensity": ring_median,
                "log1p_area": float(math.log1p(identity.area)),
                "border_distance": float(border_distance),
                "local_robust_scr": scr,
                "local_ring_robust_dispersion": ring_dispersion,
            }
        )
    return rows


def join_fixed_outcomes(
    feature_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join prediction-free rows to exactly three fixed-epoch target outcomes."""

    features: dict[str, Mapping[str, Any]] = {}
    for row in feature_rows:
        key = row.get("stable_target_id")
        if not isinstance(key, str) or not key:
            raise DifficultyAuditError("feature row lacks stable target identity")
        if key in features:
            raise DifficultyAuditError("duplicate feature stable target identity")
        features[key] = row

    outcomes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        if row.get("row_kind") != "target":
            continue
        if row.get("checkpoint", {}).get("policy") != "fixed_epoch":
            raise DifficultyAuditError("E-1b accepts only fixed_epoch target rows")
        key = row.get("stable_target_id")
        if not isinstance(key, str) or not key:
            raise DifficultyAuditError("ledger row lacks stable target identity")
        outcomes[key].append(row)
    if set(features) != set(outcomes):
        raise DifficultyAuditError("feature and fixed-ledger target universes differ")

    joined: list[dict[str, Any]] = []
    for key in sorted(features):
        group = outcomes[key]
        if len(group) != 3 or len({int(row["seed"]) for row in group}) != 3:
            raise DifficultyAuditError("each fixed target must have exactly three seeds")
        first = group[0]
        assertion_fields = (
            "dataset",
            "image_name",
            "area",
            "component_mask_sha256",
            "label_mask_sha256",
            "miss_count",
        )
        if any(
            row.get(field) != first.get(field)
            for row in group[1:]
            for field in assertion_fields
        ):
            raise DifficultyAuditError("fixed target assertion metadata drifted")
        miss_count = sum(not bool(row.get("matched")) for row in group)
        if miss_count != int(first.get("miss_count")):
            raise DifficultyAuditError("stored miss_count disagrees with seed rows")
        feature = dict(features[key])
        for field in (
            "dataset",
            "image_name",
            "area",
            "component_mask_sha256",
            "label_mask_sha256",
        ):
            if feature.get(field) != first.get(field):
                raise DifficultyAuditError(
                    f"prediction-free feature disagrees with ledger on {field}"
                )
        feature.update(
            {
                "miss_count": miss_count,
                "miss_any_seed": int(miss_count > 0),
                "miss_three_of_three": int(miss_count == 3),
                "miss_seed_ids": sorted(
                    int(row["seed"]) for row in group if not bool(row["matched"])
                ),
            }
        )
        joined.append(feature)
    return joined


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    response: str,
) -> tuple[np.ndarray, np.ndarray]:
    if response not in RESPONSES:
        raise DifficultyAuditError(f"unknown response {response}")
    values: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        covariates = [row.get(name) for name in COVARIATES]
        if any(value is None for value in covariates):
            continue
        numeric = [float(value) for value in covariates]
        if not bool(np.isfinite(numeric).all()):
            raise DifficultyAuditError("non-finite complete-case covariate")
        label = row.get(response)
        if label not in (0, 1, False, True):
            raise DifficultyAuditError("response must be binary")
        values.append(numeric)
        labels.append(int(label))
    if not values:
        raise DifficultyAuditError("no complete cases are available")
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _rank_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if not n_pos or not n_neg:
        return None
    ranks = stats.rankdata(scores, method="average")
    return float(
        (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    n_pos = int(np.sum(y == 1))
    if not n_pos or not int(np.sum(y == 0)):
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float(np.sum(precision[sorted_y == 1]) / n_pos)


def _fit_l2_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise DifficultyAuditError("invalid L2 logistic inputs")
    if len(np.unique(y)) != 2:
        raise DifficultyAuditError("L2 logistic training labels need two classes")
    design = np.column_stack((np.ones(x.shape[0]), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta
        value = float(np.sum(np.logaddexp(0.0, linear) - y * linear))
        value += float(0.5 / LODO_L2_C * np.dot(beta[1:], beta[1:]))
        gradient = design.T @ (expit(linear) - y)
        gradient[1:] += beta[1:] / LODO_L2_C
        return value, gradient

    result = optimize.minimize(
        lambda beta: objective(beta),
        np.zeros(design.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not bool(np.isfinite(result.x).all()):
        raise DifficultyAuditError(f"fixed L2 logistic failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "coefficients": [float(value) for value in result.x[1:]],
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "optimizer": "scipy_L-BFGS-B",
        "penalty": "0.5/C * sum(beta_j^2), intercept unpenalized",
        "C": LODO_L2_C,
    }


def _separation_status(design: np.ndarray, y: np.ndarray) -> str | None:
    """Detect complete or quasi-complete separation by linear feasibility.

    Complete separation has every signed margin strictly positive.  Quasi
    separation permits zero margins but requires at least one positive margin.
    The split-variable L1 bound makes the latter test bounded and deterministic.
    """

    signed_design = np.where(y[:, None] == 1, design, -design)
    complete = optimize.linprog(
        np.zeros(design.shape[1]),
        A_ub=-signed_design,
        b_ub=-np.ones(len(y)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    if complete.success:
        return "complete_separation"
    split_design = np.column_stack((signed_design, -signed_design))
    objective = -np.sum(split_design, axis=0)
    quasi = optimize.linprog(
        objective,
        A_ub=np.vstack((-split_design, np.ones((1, split_design.shape[1])))),
        b_ub=np.concatenate((np.zeros(len(y)), np.ones(1))),
        bounds=[(0.0, None)] * split_design.shape[1],
        method="highs",
    )
    if not quasi.success:
        raise DifficultyAuditError("quasi-separation LP failed")
    maximum_signed_margin_sum = float(-quasi.fun)
    if maximum_signed_margin_sum > 1e-9:
        return "quasi_complete_separation"
    return None


def lodo_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    response: str,
) -> dict[str, Any]:
    """Run fixed, untuned leave-one-dataset-out L2 logistic prediction."""

    datasets = sorted({str(row.get("dataset")) for row in rows})
    fold_records: list[dict[str, Any]] = []
    for held_out in datasets:
        train_rows = [row for row in rows if row.get("dataset") != held_out]
        test_rows = [row for row in rows if row.get("dataset") == held_out]
        train_complete = [
            row
            for row in train_rows
            if all(row.get(name) is not None for name in COVARIATES)
        ]
        test_complete = [
            row
            for row in test_rows
            if all(row.get(name) is not None for name in COVARIATES)
        ]

        def class_image_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
            return {
                "positive": len(
                    {
                        (str(row["dataset"]), str(row["image_name"]))
                        for row in values
                        if int(row[response]) == 1
                    }
                ),
                "negative": len(
                    {
                        (str(row["dataset"]), str(row["image_name"]))
                        for row in values
                        if int(row[response]) == 0
                    }
                ),
            }

        train_clusters = class_image_counts(train_complete)
        test_clusters = class_image_counts(test_complete)
        eligibility = all(
            counts[label] >= MIN_CLASS_IMAGE_CLUSTERS
            for counts in (train_clusters, test_clusters)
            for label in ("positive", "negative")
        )
        record: dict[str, Any] = {
            "held_out_dataset": held_out,
            "training_target_count": len(train_complete),
            "held_out_target_count": len(test_complete),
            "training_class_image_clusters": train_clusters,
            "held_out_class_image_clusters": test_clusters,
            "minimum_class_image_clusters": MIN_CLASS_IMAGE_CLUSTERS,
            "eligible": eligibility,
        }
        if not eligibility:
            record.update(
                {
                    "status": "ineligible_class_image_cluster_support",
                    "auroc": None,
                    "average_precision": None,
                    "model": None,
                }
            )
            fold_records.append(record)
            continue
        x_train, y_train = _matrix(train_complete, response=response)
        x_test, y_test = _matrix(test_complete, response=response)
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0, ddof=0)
        if bool(np.any(scale <= 0.0)):
            record.update(
                {
                    "status": "ineligible_zero_training_scale",
                    "eligible": False,
                    "auroc": None,
                    "average_precision": None,
                    "model": None,
                }
            )
            fold_records.append(record)
            continue
        standardized_train = (x_train - mean) / scale
        standardized_test = (x_test - mean) / scale
        try:
            model = _fit_l2_logistic(standardized_train, y_train)
        except DifficultyAuditError as exc:
            record.update(
                {
                    "status": "unresolved_optimizer_failure",
                    "eligible": False,
                    "auroc": None,
                    "average_precision": None,
                    "model": None,
                    "message": str(exc),
                }
            )
            fold_records.append(record)
            continue
        scores = expit(
            model["intercept"]
            + standardized_test @ np.asarray(model["coefficients"])
        )
        auroc = _rank_auc(y_test, scores)
        average_precision = _average_precision(y_test, scores)
        if auroc is None or average_precision is None:
            raise DifficultyAuditError("eligible LODO fold produced undefined metrics")
        record.update(
            {
                "status": "eligible_complete",
                "auroc": auroc,
                "average_precision": average_precision,
                "training_standardization": {
                    "covariates": list(COVARIATES),
                    "mean": [float(value) for value in mean],
                    "population_std": [float(value) for value in scale],
                },
                "model": model,
            }
        )
        fold_records.append(record)

    eligible = [record for record in fold_records if record["eligible"]]
    return {
        "response": response,
        "protocol": "leave_one_dataset_out_fixed_l2_logistic_v1",
        "covariates": list(COVARIATES),
        "folds": fold_records,
        "eligible_fold_count": len(eligible),
        "eligible_mean_auroc": (
            float(np.mean([record["auroc"] for record in eligible]))
            if eligible
            else None
        ),
        "eligible_mean_average_precision": (
            float(np.mean([record["average_precision"] for record in eligible]))
            if eligible
            else None
        ),
    }


def association_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    response: str,
) -> dict[str, Any]:
    """Fit descriptive unpenalized logit with dataset blocks and CR0 SEs."""

    complete = [
        row for row in rows if all(row.get(name) is not None for name in COVARIATES)
    ]
    x_raw, y = _matrix(complete, response=response)
    datasets = sorted({str(row["dataset"]) for row in complete})
    if len(datasets) < 2 or len(np.unique(y)) != 2:
        return {
            "response": response,
            "status": "unresolved_insufficient_classes_or_dataset_blocks",
            "coefficients": None,
        }
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0, ddof=0)
    if bool(np.any(scale <= 0.0)):
        return {
            "response": response,
            "status": "unresolved_zero_covariate_scale",
            "coefficients": None,
        }
    x_standardized = (x_raw - mean) / scale
    reference = datasets[0]
    dummy_names = [f"dataset[{name}]" for name in datasets[1:]]
    dummies = np.asarray(
        [
            [float(str(row["dataset"]) == name) for name in datasets[1:]]
            for row in complete
        ],
        dtype=np.float64,
    )
    design = np.column_stack(
        (np.ones(len(complete)), x_standardized, dummies)
    )
    separation_status = _separation_status(design, y)
    if separation_status is not None:
        return {
            "response": response,
            "status": f"{separation_status}_no_ordinary_mle",
            "coefficients": None,
            "separation_check": (
                "complete signed_margin>=1; quasi signed_margin>=0 with "
                "positive bounded-L1 total margin"
            ),
        }

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta
        return (
            float(np.sum(np.logaddexp(0.0, linear) - y * linear)),
            design.T @ (expit(linear) - y),
        )

    fitted = optimize.minimize(
        lambda beta: objective(beta),
        np.zeros(design.shape[1]),
        jac=True,
        method="BFGS",
        options={"maxiter": 5000, "gtol": 1e-8},
    )
    if not fitted.success or not bool(np.isfinite(fitted.x).all()):
        return {
            "response": response,
            "status": "ordinary_mle_failed",
            "message": str(fitted.message),
            "coefficients": None,
        }
    probability = expit(design @ fitted.x)
    hessian = design.T @ ((probability * (1.0 - probability))[:, None] * design)
    if np.linalg.matrix_rank(hessian) != hessian.shape[0]:
        return {
            "response": response,
            "status": "ordinary_mle_singular_hessian",
            "coefficients": None,
        }
    bread = np.linalg.inv(hessian)
    cluster_scores: dict[tuple[str, str], np.ndarray] = {}
    individual_scores = design * (y - probability)[:, None]
    for index, row in enumerate(complete):
        cluster = (str(row["dataset"]), str(row["image_name"]))
        cluster_scores.setdefault(cluster, np.zeros(design.shape[1]))
        cluster_scores[cluster] += individual_scores[index]
    meat = sum(
        (score[:, None] @ score[None, :]) for score in cluster_scores.values()
    )
    covariance = bread @ meat @ bread
    variance = np.diag(covariance)
    if bool(np.any(variance < -1e-12)):
        raise DifficultyAuditError("cluster-robust covariance has negative variance")
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    names = ["intercept", *COVARIATES, *dummy_names]
    coefficients = []
    for name, estimate, error in zip(names, fitted.x, standard_error):
        z_value = float(estimate / error) if error > 0 else None
        coefficients.append(
            {
                "term": name,
                "estimate": float(estimate),
                "cluster_robust_se_cr0": float(error),
                "z": z_value,
                "two_sided_normal_p": (
                    float(2.0 * stats.norm.sf(abs(z_value)))
                    if z_value is not None
                    else None
                ),
                "ci95": (
                    [float(estimate - 1.96 * error), float(estimate + 1.96 * error)]
                    if error > 0
                    else None
                ),
                "odds_ratio": float(np.exp(estimate)),
                "odds_ratio_ci95": (
                    [
                        float(np.exp(estimate - 1.96 * error)),
                        float(np.exp(estimate + 1.96 * error)),
                    ]
                    if error > 0
                    else None
                ),
            }
        )
    return {
        "response": response,
        "status": "descriptive_mle_complete",
        "interpretation": "association only; no causal attribution",
        "dataset_reference": reference,
        "complete_case_target_count": len(complete),
        "image_cluster_count": len(cluster_scores),
        "standardization": {
            "mean": [float(value) for value in mean],
            "population_std": [float(value) for value in scale],
        },
        "covariance": "image-cluster sandwich CR0",
        "coefficients": coefficients,
    }


def summarize_difficulty(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DifficultyAuditError("difficulty table cannot be empty")
    unavailable = {
        name: sum(row.get(name) is None for row in rows) for name in COVARIATES
    }
    unavailable_fraction = {
        name: count / len(rows) for name, count in unavailable.items()
    }
    unavailable_by_dataset = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if str(row["dataset"]) == dataset]
        unavailable_by_dataset[dataset] = {
            name: {
                "count": sum(row.get(name) is None for row in subset),
                "fraction": sum(row.get(name) is None for row in subset)
                / len(subset),
            }
            for name in COVARIATES
        }
    availability_unresolved = any(
        value > MAX_UNAVAILABLE_FRACTION for value in unavailable_fraction.values()
    )
    association = {
        response: association_analysis(rows, response=response)
        for response in RESPONSES
    }
    lodo = {response: lodo_analysis(rows, response=response) for response in RESPONSES}
    primary = lodo["miss_any_seed"]
    eligible_aurocs = [
        float(record["auroc"])
        for record in primary["folds"]
        if record["eligible"]
    ]
    enough_folds = len(eligible_aurocs) >= 2
    nearly_fully_explained = bool(
        enough_folds
        and float(np.mean(eligible_aurocs)) >= 0.90
        and all(value >= 0.85 for value in eligible_aurocs)
    )
    if availability_unresolved or not enough_folds:
        route = "UNRESOLVED"
    elif nearly_fully_explained:
        route = "E0_NO_GO_PREDICTION_FREE_DIFFICULTY"
    else:
        route = "E1B_PASS_FOR_E0_ROUTING"
    return {
        "schema_version": "dea.gate_e.prediction_free_difficulty_summary.v1",
        "target_count": len(rows),
        "covariates": list(COVARIATES),
        "responses": {
            response: {
                "positive_targets": sum(int(row[response]) for row in rows),
                "negative_targets": sum(1 - int(row[response]) for row in rows),
            }
            for response in RESPONSES
        },
        "availability": {
            "minimum_ring_pixels": MIN_RING_PIXELS,
            "maximum_unavailable_fraction": MAX_UNAVAILABLE_FRACTION,
            "unavailable_count": unavailable,
            "unavailable_fraction": unavailable_fraction,
            "by_dataset": unavailable_by_dataset,
            "zero_ring_dispersion_count": sum(
                row.get("local_ring_robust_dispersion") == 0.0 for row in rows
            ),
            "unresolved": availability_unresolved,
        },
        "association": association,
        "lodo": lodo,
        "routing": {
            "primary_response": "miss_any_seed",
            "eligible_fold_count": len(eligible_aurocs),
            "eligible_mean_auroc": (
                float(np.mean(eligible_aurocs)) if eligible_aurocs else None
            ),
            "all_eligible_fold_auroc_at_least_0_85": (
                all(value >= 0.85 for value in eligible_aurocs)
                if eligible_aurocs
                else None
            ),
            "near_complete_explanation_rule": (
                "eligible mean AUROC >=0.90 and every eligible fold AUROC >=0.85"
            ),
            "near_complete_explanation": nearly_fully_explained,
            "decision": route,
        },
    }
