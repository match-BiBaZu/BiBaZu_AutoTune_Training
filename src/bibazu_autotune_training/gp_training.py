"""Build a reviewed multi-part dataset and fit the physics-informed GP."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bibazu_autotune.ml.artifacts import FLIP_CLASSES, save_model
from bibazu_autotune.ml.features import context_from_snapshot

from .runner import parameters_match
from .storage import CampaignStore

MAX_TRIALS = 1_500


@dataclass(frozen=True)
class TrainingData:
    features: np.ndarray
    labels: tuple[str, ...]
    groups: tuple[str, ...]
    campaigns: tuple[str, ...]
    excluded: dict
    by_part: dict

    @property
    def counts(self):
        return Counter(self.labels)

    def validate(self):
        counts = self.counts
        if not counts["flipped"]:
            raise ValueError("Trainingsbeispiele für erfolgreiche Flips (flipped) fehlen.")
        if not (counts["not_flipped"] or counts["overflipped"]):
            raise ValueError("Trainingsbeispiele für not_flipped oder overflipped fehlen.")
        if len(self.labels) > MAX_TRIALS:
            raise ValueError(
                f"GP erlaubt höchstens {MAX_TRIALS} Versuche. Weniger Kampagnen wählen."
            )


def trial_label(record, annotation):
    if annotation and annotation.get("excluded"):
        return None, "Manuell ausgeschlossen"
    if record.get("status") not in {"completed", "failed"}:
        return None, "Versuch abgebrochen oder noch offen"
    if annotation:
        label = annotation.get("label")
        if label in FLIP_CLASSES:
            return label, None
        return None, "Keine gültige Flipklasse"
    evaluation = record.get("evaluation", {})
    if (
        record["status"] == "completed"
        and evaluation.get("quality") == "valid"
        and evaluation.get("label") in FLIP_CLASSES
    ):
        return evaluation["label"], None
    return None, "Keine gültige Flipklasse"


def read_campaigns(paths):
    rows, labels, groups = [], [], []
    seen, used = set(), []
    exclusions = Counter()
    by_part = {}
    for path in paths:
        try:
            store = CampaignStore(Path(path))
            trials = store.trials()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Kampagne {path} kann nicht gelesen werden: {exc}") from exc
        if store.campaign_id in seen:
            continue
        seen.add(store.campaign_id)
        metadata = store.manifest.get("metadata") or {}
        camera = metadata.get("camera") or {}
        machine = (metadata.get("plc") or {}).get("machine_metadata") or {}
        if camera.get("demo") or machine.get("simulated"):
            exclusions["Simulation"] += len(trials)
            continue
        snapshot = store.plan.snapshot
        part = snapshot.get("mesh_sha256")
        if not isinstance(part, str) or not part:
            exclusions["STL-Kennung fehlt"] += len(trials)
            continue
        part_name = store.plan.context.get("part_name") or part[:12]
        used_before = len(rows)
        references = {}
        for record in trials:
            label, reason = trial_label(record, store.annotation(record["trial_id"]))
            if reason:
                exclusions[reason] += 1
                continue
            requested = record.get("requested_parameters")
            actual = record.get("readback_parameters")
            if not actual or not parameters_match(actual, requested):
                exclusions["SPS-Parameter fehlen oder weichen ab"] += 1
                continue
            array_index = store.plan.array_index
            try:
                array = requested["arrays"][array_index - 1]
                duration = array["pulse_duration_ms"]
                # A campaign fixes the hardware setup and varies pressure and duration only.
                if duration not in references:
                    context = context_from_snapshot(snapshot, requested, array_index)
                    references[duration] = context, context.analytical_gauge(duration)
                context, analytical = references[duration]
                rows.append(context.row(array["pressure_mbar"], duration, analytical))
            except (ValueError, TypeError, OverflowError, KeyError, IndexError) as exc:
                exclusions[f"Merkmale unvollständig: {exc}"] += 1
                continue
            labels.append(label)
            groups.append(part)
            by_part.setdefault(part, {"name": part_name, "counts": Counter()})["counts"][label] += 1
        if len(rows) > used_before:
            used.append(store.campaign_id)
    return TrainingData(
        np.asarray(rows, dtype=float),
        tuple(labels),
        tuple(groups),
        tuple(used),
        dict(exclusions),
        by_part,
    )


def create_gp():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        GaussianProcessClassifier(
            kernel=ConstantKernel(1.0, (0.01, 100)) * RBF(1.0, (0.01, 100)),
            multi_class="one_vs_rest",
            random_state=0,
            n_jobs=1,
        ),
    )


def validation_folds(features, labels, groups):
    from sklearn.model_selection import GroupKFold

    count = len(set(groups))
    if count < 2:
        return []
    folds = list(GroupKFold(n_splits=min(count, 5)).split(features, labels, groups))
    classes = set(labels)
    if any(set(labels[train]) != classes for train, _ in folds):
        return []
    return folds


def validate_on_parts(features, labels, groups, progress):
    from sklearn.metrics import confusion_matrix, log_loss

    folds = validation_folds(features, labels, groups)
    if not folds:
        return {"status": "unvalidated", "folds": 0}

    ordered = sorted(set(labels))
    predictions = np.empty((len(labels), len(ordered)))
    for i, (train, test) in enumerate(folds, 1):
        progress(f"Bauteilprüfung {i}/{len(folds)} …")
        model = create_gp().fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])

    def metrics(y, p):
        return {
            "trials": len(y),
            "log_loss": float(log_loss(y, p, labels=ordered)),
            "confusion_matrix": confusion_matrix(
                y, np.asarray(ordered)[p.argmax(axis=1)], labels=ordered
            ).tolist(),
        }

    parts = np.asarray(groups)
    by_part = {
        part: metrics(labels[parts == part], predictions[parts == part])
        for part in sorted(set(parts))
    }
    return {
        "status": "validated",
        "folds": len(folds),
        "class_order": ordered,
        **metrics(labels, predictions),
        "by_part": by_part,
    }


def train_and_save(dataset, name, directory=None, progress=lambda _message: None):
    from threadpoolctl import threadpool_limits

    dataset.validate()
    features, labels = dataset.features, np.asarray(dataset.labels)
    with threadpool_limits(limits=1):
        validation = validate_on_parts(features, labels, dataset.groups, progress)
        progress("Trainiere GP auf allen ausgewählten Versuchen …")
        fitted = create_gp().fit(features, labels)
    progress("Modell wird gespeichert …")
    return save_model(
        fitted,
        {
            "name": name,
            "campaign_ids": dataset.campaigns,
            "part_groups": sorted(set(dataset.groups)),
            "class_counts": dict(dataset.counts),
            "trials": len(labels),
            "feature_bounds": np.column_stack(
                [features.min(axis=0), features.max(axis=0)]
            ).tolist(),
            "validation": validation,
        },
        directory,
    )
