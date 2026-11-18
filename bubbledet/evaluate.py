"""Match detections to ground truth and compute precision/recall per class."""
from __future__ import annotations

from dataclasses import dataclass

from .models import BUBBLE, PARTICLE, DetectedObject, GroundTruthObject


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    support: int  # number of ground-truth objects of this class


@dataclass(frozen=True)
class EvaluationResult:
    detection_recall: float          # fraction of GT objects detected at all
    matched: int
    total_truth: int
    per_class: dict[str, ClassMetrics]
    overall_accuracy: float          # classification accuracy over matched pairs

    def to_text(self) -> str:
        lines = [
            "Evaluation",
            "----------",
            f"Detection recall : {self.detection_recall:.1%} "
            f"({self.matched}/{self.total_truth} ground-truth objects found)",
            f"Classification accuracy (matched): {self.overall_accuracy:.1%}",
            "",
            f"{'class':<10}{'precision':>12}{'recall':>10}{'support':>10}",
        ]
        for name, m in self.per_class.items():
            lines.append(
                f"{name:<10}{m.precision:>12.1%}{m.recall:>10.1%}{m.support:>10d}"
            )
        return "\n".join(lines)


def _match(objects, truth):
    """Greedy nearest-centroid matching; a GT matches the closest unused object
    within its own radius. Returns list of (object, gt)."""
    used = [False] * len(objects)
    pairs = []
    for gt in truth:
        tol = gt.diameter_px  # generous: one full diameter
        best_i = -1
        best_d = tol
        for i, o in enumerate(objects):
            if used[i]:
                continue
            d = ((o.centroid[0] - gt.cx) ** 2 + (o.centroid[1] - gt.cy) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            pairs.append((objects[best_i], gt))
    return pairs


def evaluate(
    objects: list[DetectedObject], truth: list[GroundTruthObject]
) -> EvaluationResult:
    pairs = _match(objects, truth)
    matched = len(pairs)
    total = len(truth)
    detection_recall = matched / total if total else 0.0

    per_class: dict[str, ClassMetrics] = {}
    correct = sum(1 for o, gt in pairs if o.label == gt.kind)
    overall_acc = correct / matched if matched else 0.0

    for cls in (PARTICLE, BUBBLE):
        tp = sum(1 for o, gt in pairs if o.label == cls and gt.kind == cls)
        fp = sum(1 for o, gt in pairs if o.label == cls and gt.kind != cls)
        support = sum(1 for gt in truth if gt.kind == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        per_class[cls] = ClassMetrics(precision, recall, support)

    return EvaluationResult(
        detection_recall=detection_recall,
        matched=matched,
        total_truth=total,
        per_class=per_class,
        overall_accuracy=overall_acc,
    )
