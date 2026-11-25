"""Exercise stored campaigns, whole-workpiece validation and AutoTune inference."""

import hashlib
import json
from dataclasses import asdict, replace
from uuid import uuid4

import numpy as np
import pytest
from bibazu_autotune.analytical_model import ModelConfiguration
from bibazu_autotune.ml.artifacts import FLIP_CLASSES, load_model, read_metadata
from bibazu_autotune.ml.features import context_from_snapshot
from bibazu_autotune.ml.recommendation import RecommendationDialog
from bibazu_autotune.ml.search import GridRange, score_grid
from test_experiments import plan_assets as plan_assets
from test_experiments_gui import training_window as training_window

from bibazu_autotune_training.gp_training import (
    create_gp,
    read_campaigns,
    train_and_save,
    validation_folds,
)
from bibazu_autotune_training.models import SweepRange
from bibazu_autotune_training.storage import CampaignStore


def campaign_for_part(
    tmp_path, plan, index, metadata=None, *, outcomes=FLIP_CLASSES, pressure=None
):
    mesh = tmp_path / f"part-{index}.stl"
    mesh.write_text(f"distinct geometry fixture {index}", encoding="utf-8")
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    snapshot = {
        "mesh_sha256": digest,
        "configuration": asdict(ModelConfiguration()),
        "material": {"mass_kg": 0.025 + index * 0.005},
        "candidate": {
            "mechanics": {"inertia_axis_kg_m2": 1e-5 + index * 1e-6, "delta_h_m": 0.007},
            "impact": {"lever_arm_m": 0.020 + index * 0.001, "area_eff_m2": 1e-5},
            "turn": {"signed_angle_deg": 90},
        },
    }
    plan = replace(
        plan,
        context={**plan.context, "mesh_sha256": digest, "part_name": f"Part {index}"},
        snapshot=snapshot,
        pressure=pressure or SweepRange(1000, 3000, 1000),
        pulse=SweepRange(100, 100, 10),
    )
    store = CampaignStore.create(
        tmp_path / "campaigns", plan, assets={"mesh": mesh}, metadata=metadata
    )
    for point, label in zip(plan.points(), outcomes, strict=True):
        save_example(store, point, label)
    return store


def save_example(store, point, label, *, confirmed=True, reviewed=False, excluded=False):
    parameters = store.plan.parameters_at(point)
    trial_id = uuid4().hex
    store.save_trial(
        {
            "trial_id": trial_id,
            "created_utc": "2026-11-25T12:00:00Z",
            "requested_parameters": parameters,
            "readback_parameters": parameters if confirmed else {},
            "status": "failed" if reviewed else "completed",
            "evaluation": {
                "quality": "review_required" if reviewed else "valid",
                "label": None if reviewed else label,
                "rotation_deg": None,
                "max_rotation_deg": None,
            },
        }
    )
    if reviewed or excluded:
        store.annotate(trial_id, label, excluded, "Im Prüfvideo beurteilt")


def test_train_four_parts_and_use_the_model_on_another_geometry(plan_assets, tmp_path):
    plan, _ = plan_assets
    stores = [campaign_for_part(tmp_path, plan, index) for index in range(3)]
    stores.append(
        campaign_for_part(tmp_path, plan, 3, outcomes=("not_flipped", "flipped", "flipped"))
    )
    paths = [store.directory for store in stores]
    data = read_campaigns(paths + paths[:1])
    assert data.counts == {"not_flipped": 4, "flipped": 5, "overflipped": 3}
    assert len(data.groups) == 12 and len(data.campaigns) == 4
    groups = np.asarray(data.groups)
    folds = validation_folds(data.features, np.asarray(data.labels), groups)
    assert len(folds) == 4
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
        assert len(set(groups[test])) == 1

    artifact = train_and_save(data, "Vier Geometrien", tmp_path / "models")
    loaded = load_model(artifact)
    assert (artifact / "model.skops").is_file()
    assert loaded.metadata["schema_version"] == 1
    assert loaded.metadata["validation"]["folds"] == 4
    assert len(loaded.metadata["validation"]["by_part"]) == 4
    direct = create_gp().fit(data.features, data.labels)
    np.testing.assert_allclose(
        loaded.probabilities(data.features),
        direct.predict_proba(data.features)[
            :, [list(direct.classes_).index(c) for c in FLIP_CLASSES]
        ],
    )

    new = campaign_for_part(tmp_path, plan, 4)
    assert new.plan.snapshot["mesh_sha256"] not in loaded.metadata["part_groups"]
    context = context_from_snapshot(new.plan.snapshot, new.plan.parameters, new.plan.array_index)
    grid = score_grid(loaded, context, GridRange(1000, 3000, 1000), GridRange(100, 120, 10))
    assert len(grid) == 9
    assert grid[0].flip_probability == max(point.flip_probability for point in grid)
    assert "mass_kg" in grid[0].outside_features
    assert grid[0].analytical_mbar == pytest.approx(
        context.analytical_gauge(grid[0].pulse_duration_ms)
    )
    metadata = read_metadata(artifact)
    metadata["feature_version"] = "unknown"
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="anderen Merkmalsatz"):
        load_model(artifact)


def test_reviewed_labels_without_angles_and_missing_class_detection(plan_assets, tmp_path):
    plan, _ = plan_assets
    source = campaign_for_part(tmp_path, plan, 0)
    simulated = campaign_for_part(tmp_path, plan, 1, metadata={"camera": {"demo": True}})
    point = source.plan.points()[1]
    save_example(source, point, "flipped", reviewed=True)
    save_example(source, point, "flipped", reviewed=True, confirmed=False)
    save_example(source, point, "flipped", excluded=True)
    data = read_campaigns([source.directory, simulated.directory])
    assert data.counts == {"not_flipped": 1, "flipped": 2, "overflipped": 1}
    assert data.excluded == {
        "Simulation": 3,
        "Manuell ausgeschlossen": 1,
        "SPS-Parameter fehlen oder weichen ab": 1,
    }
    with pytest.raises(ValueError, match="erfolgreiche Flips"):
        replace(data, labels=("not_flipped", "overflipped")).validate()
    with pytest.raises(ValueError, match="not_flipped oder overflipped"):
        replace(data, labels=("flipped", "flipped")).validate()
    path = train_and_save(data, "Ein Bauteil", tmp_path / "models")
    assert read_metadata(path)["validation"]["status"] == "unvalidated"


@pytest.mark.parametrize(
    "outcomes,missing",
    [
        (("not_flipped", "flipped", "flipped"), "overflipped"),
        (("flipped", "flipped", "overflipped"), "not_flipped"),
    ],
)
def test_training_page_exports_two_class_model_for_recommendation_dialog(
    training_window,
    plan_assets,
    tmp_path,
    outcomes,
    missing,
):
    window, wait, errors = training_window
    plan, _ = plan_assets
    grid = SweepRange(2000, 6000, 2000)
    sources = [
        campaign_for_part(tmp_path, plan, i, outcomes=outcomes, pressure=grid) for i in range(2)
    ]
    target = campaign_for_part(tmp_path, plan, 2)
    page = window.training
    page.add_campaigns([source.directory for source in sources])
    page.inspect_button.click()
    wait(lambda: page.worker is None)
    assert "6 verwendbare Versuche" in page.report.toPlainText()
    assert f"Keine Schätzung für {missing}" in page.status.text()
    assert page.train_button.isEnabled()
    page.train_button.click()
    wait(lambda: page.worker is None)
    assert "Modell gespeichert" in page.status.text()

    context = context_from_snapshot(
        target.plan.snapshot, target.plan.parameters, target.plan.array_index
    )
    dialog = RecommendationDialog(
        context, target.plan.snapshot["mesh_sha256"], "Neues Bauteil", 2000, 100, window
    )
    try:
        assert dialog.models.count() == 1
        model = load_model(dialog.models.currentData())
        assert model.classes == tuple(label for label in FLIP_CLASSES if label != missing)
        assert model.metadata["validation"]["status"] == "validated"
        assert len(model.metadata["validation"]["confusion_matrix"]) == 2
        dialog.pressure[1].setValue(6000)
        dialog.pressure[2].setValue(2000)
        dialog.search.click()
        wait(lambda: dialog.worker is None)
        assert dialog.grid.rowCount() == 3
        assert "Vorschlag" in dialog.status.text()
        assert "Extrapolation" in dialog.status.text()
        assert "Bauteil war nicht in den Trainingsdaten" in dialog.status.text()
        assert f"keine Schätzung für {missing}" in dialog.status.text()
        assert missing not in dialog.grid.rows[0].probabilities
        assert dialog.grid.data(dialog.grid.index(0, 3 + FLIP_CLASSES.index(missing))) == "—"
        dialog.duration[1].setValue(110)
        assert dialog.grid.rowCount() == 0
    finally:
        if dialog.worker is not None:
            wait(lambda: dialog.worker is None)
        dialog.close()
    assert not errors
