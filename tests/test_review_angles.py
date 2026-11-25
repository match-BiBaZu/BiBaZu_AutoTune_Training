"""Manual interval outcomes retain provenance, original data, class labels and actual gaps."""

import math
import os
from dataclasses import replace
from uuid import uuid4

import pytest
from test_experiments import plan_assets as plan_assets
from test_targeted_classification import measurements

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bibazu_autotune.engine import atomic_json
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QPushButton

from bibazu_autotune_training.classification import evaluate
from bibazu_autotune_training.dataset import DatasetReader
from bibazu_autotune_training.review import AngleReviewDialog, ReviewPage
from bibazu_autotune_training.review_angles import interval_evaluation, suggested_interval
from bibazu_autotune_training.storage import CampaignStore, read_json, write_samples


@pytest.fixture
def angle_trial(plan_assets, tmp_path):
    plan, assets = plan_assets
    store = CampaignStore.create(tmp_path / "campaigns", plan, assets=assets)
    trial_id = uuid4().hex
    rows = measurements(0)
    for index in range(11, 27):
        rows[index]["angle_deg"] = 45 * math.sin(math.pi * (index - 10) / 17)
    last = len(rows) - 1
    for index in range(20):
        rows.append(
            {**rows[-1], "timestamp": (last + index + 1) * 0.01, "frame_id": 1000 + index * 3}
        )
    for _ in range(12):
        rows.append(
            {
                **rows[-1],
                "timestamp": rows[-1]["timestamp"] + 0.01,
                "frame_id": rows[-1]["frame_id"] + 1,
            }
        )
    evaluation, samples = evaluate(rows, plan.classification)
    assert evaluation["rotation_deg"] is None
    record = {
        "trial_id": trial_id,
        "slot": 0,
        "point_index": 0,
        "repetition": 1,
        "created_utc": "2026-11-25",
        "requested_parameters": plan.parameters_at(plan.points()[0]),
        "status": "completed",
        "resolution": "uncertain",
        "evaluation": evaluation,
        "clip": "clip",
    }
    store.save_trial(record)
    directory = store.trial_directory(trial_id)
    (directory / "clip").mkdir()
    manifest = {
        "schema_version": 2,
        "metadata": {"run_id": store.campaign_id, "trial_id": trial_id},
        "frame_count": len(rows),
        "frames": [
            {"timestamp": row["timestamp"], "frame_id": row["frame_id"], "measurement": row}
            for row in rows
        ],
        "media": {"file": "review.mp4", "state": "deleted"},
    }
    atomic_json(directory / "clip/manifest.json", manifest)
    write_samples(directory / "measurements.csv", samples)
    store.annotate(trial_id, "not_flipped", False, "Flipklasse im Video geprüft")
    return store, trial_id, rows, last


def test_missing_angles_do_not_exclude_reviewed_class_and_interval_recovers_measured_outcomes(
    angle_trial,
):
    store, trial_id, rows, last = angle_trial
    directory = store.trial_directory(trial_id)
    original = {
        name: (directory / name).read_bytes()
        for name in ("trial.json", "measurements.csv", "annotations.json", "clip/manifest.json")
    }
    before = DatasetReader([store]).classifier_rows()
    assert len(before) == 1 and before[0]["outcome_rotation_deg"] is None
    assert suggested_interval(rows, store.plan.classification) == (0, last)
    store.save_angle_review(trial_id, 0, last, "Vollständige Bewegung geprüft")
    row = DatasetReader([store]).classifier_rows()[0]
    assert row["label"] == "not_flipped"
    assert row["outcome_rotation_deg"] == pytest.approx(0)
    assert row["outcome_max_rotation_deg"] > 44
    assert row["automatic_outcome_rotation_deg"] is None
    assert row["outcome_source"] == "reviewed_interval"
    assert (row["outcome_first_frame"], row["outcome_last_frame"]) == (1, last + 1)
    assert all((directory / name).read_bytes() == value for name, value in original.items())
    saved = (directory / "angle_reviews.json").read_bytes()
    store.save_angle_review(trial_id, 0, last, "Vollständige Bewegung geprüft")
    assert (directory / "angle_reviews.json").read_bytes() == saved
    assert CampaignStore(store.directory).angle_review(trial_id)["usable"]
    DatasetReader([store]).export_csv(directory.parent.parent.parent / "angles.csv")


@pytest.mark.parametrize("first,last", [(-1, 10), (10, 3), (0, 999), (True, 10)])
def test_invalid_interval_is_not_saved(angle_trial, first, last):
    store, trial_id, rows, final = angle_trial
    with pytest.raises(ValueError, match="Auswertebereich"):
        store.save_angle_review(trial_id, first, last, "Geprüft")
    assert not (store.trial_directory(trial_id) / "angle_reviews.json").exists()


def test_critical_gap_or_unstable_endpoint_cannot_be_filled_from_class(angle_trial):
    store, trial_id, rows, last = angle_trial
    with pytest.raises(ValueError, match="nicht zuverlässig"):
        store.save_angle_review(trial_id, 0, len(rows) - 1, "Geprüft")
    with pytest.raises(ValueError, match="nicht zuverlässig"):
        interval_evaluation(rows, store.plan.classification, 0, 20)
    assert DatasetReader([store]).rows().__next__()["outcome_rotation_deg"] is None


def test_measurement_change_invalidates_review_but_media_change_does_not(angle_trial):
    store, trial_id, rows, last = angle_trial
    store.save_angle_review(trial_id, 0, last, "Geprüft")
    path = store.trial_directory(trial_id) / "clip/manifest.json"
    manifest = read_json(path)
    manifest["media"]["state"] = "failed"
    atomic_json(path, manifest)
    assert store.angle_review(trial_id)["usable"]
    manifest["frames"][20]["measurement"]["angle_deg"] = 80
    atomic_json(path, manifest)
    assert not store.angle_review(trial_id)["usable"]
    row = DatasetReader([store]).classifier_rows()[0]
    assert row["outcome_source"] == "automatic"
    assert row["outcome_rotation_deg"] is None
    assert row["angle_review_error"]
    assert row["label"] == "not_flipped"


def test_reset_is_auditable_and_configuration_change_is_detected(angle_trial):
    store, trial_id, rows, last = angle_trial
    store.save_angle_review(trial_id, 0, last, "Geprüft")
    store.plan = replace(
        store.plan, classification=replace(store.plan.classification, camera_direction=-1)
    )
    assert not store.angle_review(trial_id)["usable"]
    store.clear_angle_review(trial_id)
    assert store.angle_review(trial_id) is None
    assert DatasetReader([store]).classifier_rows()[0]["outcome_rotation_deg"] is None
    path = store.trial_directory(trial_id) / "angle_reviews.json"
    history = read_json(path)["history"]
    assert len(history) == 2 and history[-1]["cleared"]
    store.clear_angle_review(trial_id)
    assert len(read_json(path)["history"]) == 2


def test_dialog_requires_confirmation_and_page_shows_effective_outcomes(
    angle_trial, qt_application, monkeypatch
):
    store, trial_id, rows, last = angle_trial
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    page = ReviewPage()
    page.load(store.directory)
    assert page.angle_review_button.isEnabled()  # Measurement review works after video deletion.

    def choose(dialog):
        save = dialog.buttons.button(QDialogButtonBox.StandardButton.Save)
        assert not save.isEnabled()
        assert dialog.evaluation is not None
        dialog.confirmed.setChecked(True)
        assert save.isEnabled()
        dialog.last.setValue(len(rows))
        assert not save.isEnabled() and not dialog.confirmed.isChecked()
        dialog.last.setValue(last + 1)
        assert not save.isEnabled()
        dialog.confirmed.setChecked(True)
        save.click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AngleReviewDialog, "exec", choose)
    page.angle_review_button.click()
    assert page.selected()["outcome_source"] == "reviewed_interval"
    assert "Winkelwerte aus geprüftem Bereich" in page.details.text()
    assert page.curve.interval == (rows[0]["timestamp"], rows[last]["timestamp"])
    assert page.clear_angles_button.isEnabled()
    page.clear_angles_button.click()
    assert page.selected()["outcome_source"] == "automatic"
    assert page.curve.interval is None
    assert not page.clear_angles_button.isEnabled()
    page.close()
    page.deleteLater()
    qt_application.sendPostedEvents(page, QEvent.Type.DeferredDelete)
    assert not errors


def test_manual_class_confirmation_searches_angles_and_preserves_explicit_interval(
    angle_trial, qt_application
):
    store, trial_id, rows, last = angle_trial
    directory = store.trial_directory(trial_id)
    original = (directory / "trial.json").read_bytes()
    page = ReviewPage()
    page.load(store.directory)
    assert not (directory / "angle_reviews.json").exists()  # Merely loading stays read-only.
    assert page.selected()["angle_review_required"]
    assert not any(b.text() == "Prüfung speichern" for b in page.findChildren(QPushButton))
    page.quick_review(trial_id, "not_flipped")
    row = page.selected()
    assert row["angle_values_complete"] and not row["angle_review_required"]
    assert row["outcome_max_rotation_deg"] > 44
    assert row["outcome_selection_method"] == "automatic_after_label_review"
    assert not store.angle_review(trial_id)["complete_event_confirmed"]
    path = directory / "angle_reviews.json"
    saved = path.read_bytes()
    page.quick_review(trial_id, "not_flipped")
    assert path.read_bytes() == saved
    store.save_angle_review(trial_id, 0, last - 2, "Bereich gezielt geprüft")
    manual = path.read_bytes()
    page.quick_review(trial_id, "flipped")
    assert path.read_bytes() == manual
    assert page.selected()["outcome_selection_method"] == "manual_interval"
    assert store.annotation(trial_id)["label"] == "flipped"
    assert (directory / "trial.json").read_bytes() == original
    page.close()
    page.deleteLater()
    qt_application.sendPostedEvents(page, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("case", ["during_motion", "no_baseline", "later_motion", "class_mismatch"])
def test_failed_automatic_search_marks_row_red_but_keeps_confirmed_class(
    angle_trial, qt_application, case
):
    store, trial_id, rows, last = angle_trial
    directory = store.trial_directory(trial_id)
    path = directory / "clip/manifest.json"
    manifest = read_json(path)
    if case == "during_motion":
        for frame in manifest["frames"][20:]:
            frame["frame_id"] += 100
    elif case == "no_baseline":
        for frame in manifest["frames"]:
            frame["measurement"]["angle_deg"] = None
    elif case == "later_motion":
        manifest["frames"][last + 5]["measurement"]["angle_deg"] = 30
    atomic_json(path, manifest)
    page = ReviewPage()
    page.load(store.directory)
    label = "flipped" if case == "class_mismatch" else "not_flipped"
    page.quick_review(trial_id, label)
    assert store.annotation(trial_id)["label"] == label
    assert not store.annotation(trial_id)["excluded"]
    assert page.selected()["angle_review_required"]
    assert page.selected()["outcome_rotation_deg"] is None
    assert "Winkel fehlen" in page.table.item(0, 5).text()
    assert page.table.item(0, 5).background().color().name() == "#fee2e2"
    assert "Winkelwerte fehlen" in page.details.text()
    page.filter.setCurrentText("Winkelwerte fehlen")
    assert len(page.records) == 1
    assert len(DatasetReader([store]).classifier_rows()) == 1
    saved = (directory / "angle_reviews.json").read_bytes()
    page.quick_review(trial_id, label)
    assert (directory / "angle_reviews.json").read_bytes() == saved
    page.quick_review(trial_id, None)
    assert not page.records  # An excluded class is no longer in the missing-angle filter.
    assert not DatasetReader([store]).classifier_rows()
    page.close()
    page.deleteLater()
    qt_application.sendPostedEvents(page, QEvent.Type.DeferredDelete)
