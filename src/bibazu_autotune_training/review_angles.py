"""Measured angular outcomes from an explicitly reviewed interval, never from its class."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .classification import evaluate
from .classification_v2 import delta, number, stable_window, valid


def complete_angles(evaluation):
    return all(number(evaluation.get(key)) for key in ("rotation_deg", "max_rotation_deg"))


def fingerprint(rows):
    content = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def interval_evaluation(rows, config, first, last):
    if type(first) is not int or type(last) is not int or not 0 <= first < last < len(rows):
        raise ValueError("Ungültiger Auswertebereich.")
    result, _ = evaluate(rows[first : last + 1], replace(config, algorithm_version=2))
    if result["quality"] != "valid":
        raise ValueError(
            "Auswertebereich nicht zuverlässig auswertbar: " + "; ".join(result["reasons"])
        )
    return result


def suggested_interval(rows, config):
    """Propose an observed component; the user must verify it contains the complete event."""
    _, samples = evaluate(rows, replace(config, algorithm_version=2))
    components = {}
    for index, sample in enumerate(samples):
        if sample["continuous_angle_deg"] is not None:
            components.setdefault(sample["segment"], []).append(index)
    fallback = None
    for indices in components.values():
        first, last = indices[0], indices[-1]
        try:
            result = interval_evaluation(rows, config, first, last)
        except ValueError:
            continue
        if fallback is None:
            fallback = (first, last)
        if (
            abs(result["max_rotation_deg"]) > config.stable_span_deg
            or abs(result["rotation_deg"]) > config.stable_span_deg
        ):
            return first, last
    return fallback or (0, max(0, len(rows) - 1))


def automatic_interval(rows, config):
    """Keep the first observed baseline and movement together; never choose a quiet tail."""
    _, samples = evaluate(rows, replace(config, algorithm_version=2))
    initial = next(
        (
            i
            for i in range(len(samples))
            if stable_window(samples, i, 1, config.stable_seconds, config.stable_span_deg)
        ),
        None,
    )
    if initial is None:
        raise ValueError("Kein Messbereich mit stabiler Ausgangslage gefunden.")
    if any(valid(row) for row in rows[:initial]):
        raise ValueError("Keine durchgehend beobachtete Ausgangslage vor der Bewegung.")
    segment = samples[initial]["segment"]
    indices = [
        i
        for i in range(initial, len(samples))
        if samples[i]["segment"] == segment and samples[i]["continuous_angle_deg"] is not None
    ]
    last = indices[-1]
    result = interval_evaluation(rows, config, initial, last)
    if (
        stable_window(samples, last, -1, max(0.2, config.stable_seconds), config.stable_span_deg)
        is None
    ):
        raise ValueError("Kein Messbereich mit mindestens 200 ms stabiler Endlage gefunden.")
    if any(
        abs(delta(result["final_angle_deg"], row["angle_deg"])) > min(5.0, config.stable_span_deg)
        for row in rows[last + 1 :]
        if valid(row)
    ):
        raise ValueError(
            "Weitere Bewegung nach dem gefundenen Bereich; vollständiger Flip nicht gesichert."
        )
    return initial, last
