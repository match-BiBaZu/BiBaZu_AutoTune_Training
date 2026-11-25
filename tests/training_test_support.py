"""Local geometry fixtures; training tests do not import the AutoTune test suite."""

import math

import trimesh
from bibazu_autotune.trial_context import manual_context
from bibazu_autotune.workpiece import WorkpieceSelection


def write_box(path, *, ascii=False, translation=(0, 0, 0)):
    mesh = trimesh.creation.box(extents=[20, 30, 40])
    mesh.apply_translation(translation)
    if ascii:
        path.write_text(trimesh.exchange.stl.export_stl_ascii(mesh), "utf-8")
    else:
        path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return path


def pose(axis="y", angle=90, sha="a" * 64):
    target = [0.0, 0.0, 0.0, math.cos(math.radians(angle) / 2)]
    target["xyz".index(axis)] = math.sin(math.radians(angle) / 2)
    return manual_context("Box", sha, [0, 0, 0, 1], target)


def select_manual(window, analysis, axis="y", angle=90):
    value = pose(axis, angle, analysis.sha256)
    context = manual_context(
        analysis.path.stem,
        analysis.sha256,
        value["source_quaternion_xyzw"],
        value["target_quaternion_xyzw"],
        rotation_axis=axis,
        signed_angle_deg=angle,
    )
    window.accept_selection(WorkpieceSelection(context, analysis.path, analysis.unit, analysis))


def prepare(window, wait, tmp_path, axis="y", angle=90):
    dialog = window.autotune_settings
    dialog.set_mesh(write_box(tmp_path / "box.stl"))
    dialog.density.setText("2700")
    wait(lambda: dialog.analysis is not None)
    select_manual(window, dialog.analysis, axis, angle)
    panel = dialog.model_panel
    # These are regression cases for the earlier nozzle geometry.
    panel.fields["diameter_m"][0].setValue(0.7)
    panel.fields["pitch_m"][0].setValue((24 - 0.7) / 7)
    panel.fields["spread_angle_deg"][0].setValue(0)
    panel.fields["offset_safety_m"][0].setValue(0)
    panel.offset_strategy.setCurrentIndex(panel.offset_strategy.findData("fixed"))
    panel.fields["uphill_deg"][0].setValue(0)
    panel.fields["cross_slope_deg"][0].setValue(0)
    for mount, _, _, center in panel.mount_rows:
        center.setValue(
            (15 if mount.normal_axis == "z" else 20)
            + (200 * (mount.nozzle_index - 1) if mount.nozzle_index <= 3 else 0)
        )
        if mount.nozzle_index >= 5:
            panel.longitudinal_fields[(mount.array_index, mount.nozzle_index)].setText(
                str(200 * (mount.nozzle_index - 4))
            )
    for row in window.editor.rows:
        row["pulse_duration_ms"].setValue(500)
    return dialog, panel
