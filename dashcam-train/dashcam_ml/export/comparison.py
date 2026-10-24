"""Pure metrics comparing Core ML output with the PyTorch reference (unit-testable)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dashcam_ml.domain.entities import Detection


@dataclass(frozen=True, slots=True)
class DetectionComparison:
    reference_count: int
    candidate_count: int
    matched_count: int
    mean_iou: float
    mean_confidence_abs_diff: float


@dataclass(frozen=True, slots=True)
class MaskComparison:
    mean_abs_prob_diff: float
    mask_iou: float


def compare_detections(
    *, reference: Sequence[Detection], candidate: Sequence[Detection], match_iou_threshold: float
) -> DetectionComparison:
    """Greedy one-to-one matching of same-class boxes by IoU."""
    pairs: list[tuple[float, int, int]] = []
    for reference_index, reference_detection in enumerate(reference):
        for candidate_index, candidate_detection in enumerate(candidate):
            if reference_detection.class_name != candidate_detection.class_name:
                continue
            overlap: float = reference_detection.box.iou(other=candidate_detection.box)
            if overlap >= match_iou_threshold:
                pairs.append((overlap, reference_index, candidate_index))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    ious: list[float] = []
    confidence_diffs: list[float] = []
    for overlap, reference_index, candidate_index in pairs:
        if reference_index in used_reference or candidate_index in used_candidate:
            continue
        used_reference.add(reference_index)
        used_candidate.add(candidate_index)
        ious.append(overlap)
        confidence_diffs.append(abs(reference[reference_index].confidence - candidate[candidate_index].confidence))
    return DetectionComparison(
        reference_count=len(reference),
        candidate_count=len(candidate),
        matched_count=len(ious),
        mean_iou=float(np.mean(a=ious)) if ious else 0.0,
        mean_confidence_abs_diff=float(np.mean(a=confidence_diffs)) if confidence_diffs else 0.0,
    )


def compare_probability_maps(
    *, reference: npt.NDArray[np.float32], candidate: npt.NDArray[np.float32], threshold: float
) -> MaskComparison:
    reference_mask: npt.NDArray[np.bool_] = reference > threshold
    candidate_mask: npt.NDArray[np.bool_] = candidate > threshold
    union: int = int(np.logical_or(reference_mask, candidate_mask).sum())
    intersection: int = int(np.logical_and(reference_mask, candidate_mask).sum())
    return MaskComparison(
        mean_abs_prob_diff=float(np.mean(a=np.abs(reference - candidate))),
        mask_iou=intersection / union if union > 0 else 1.0,
    )
