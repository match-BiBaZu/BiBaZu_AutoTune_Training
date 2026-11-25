"""Installed entry points, shared service ownership and repository-independent startup."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from bibazu_autotune import app as autotune_app
from bibazu_autotune import pressure_service
from bibazu_autotune.analytical_model import AnalyticalPressureModel
from bibazu_autotune.clients import PressureClient, VisionClient

from bibazu_autotune_training import app, paths


@pytest.mark.parametrize("mode", ["module", "console"])
def test_installed_help_works_outside_repository(tmp_path, mode):
    command = [sys.executable, "-I", "-m", "bibazu_autotune_training", "--help"]
    if mode == "console":
        suffix = ".exe" if sys.platform == "win32" else ""
        command = [
            str(Path(sys.executable).parent / ("bibazu-autotune-training" + suffix)),
            "--help",
        ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--pressure-url" in result.stdout and "--demo" in result.stdout
    assert "--plc-ip" in result.stdout and "--output-dir" in result.stdout
    assert not list(tmp_path.iterdir())


def test_model_editor_and_clients_are_shared_with_autotune():
    assert app.ParameterEditor is autotune_app.ParameterEditor
    assert app.AnalyticalPressureModel is AnalyticalPressureModel
    assert app.PressureClient is PressureClient
    assert app.VisionClient is VisionClient


@pytest.mark.parametrize(
    "arguments,bundled",
    [([], True), (["--demo"], False), (["--pressure-url", "http://127.0.0.1:5555"], False)],
)
def test_startup_retains_bundled_service_and_explicit_modes(monkeypatch, arguments, bundled):
    calls = []

    class FakeApplication:
        def __init__(self, args):
            pass

        def exec(self):
            return 0

    class FakeWindow:
        def __init__(self, pressure_url, vision_url, output_dir, *, demo):
            calls.append(("window", pressure_url, vision_url, output_dir, demo))

        def show(self):
            pass

    class FakeService:
        def stop(self):
            calls.append("service stopped")

    class FakeServer:
        port = 54321

        def stop(self):
            calls.append("server stopped")

    def start(args):
        calls.append("bundled service")
        return FakeService(), FakeServer()

    monkeypatch.setattr(pressure_service, "start_bundled", start)
    monkeypatch.setattr(app, "QApplication", FakeApplication)
    monkeypatch.setattr(app, "TrainingWindow", FakeWindow)
    monkeypatch.setattr(sys, "argv", ["training", *arguments])
    assert app.main() == 0
    assert ("bundled service" in calls) == bundled
    window = next(item for item in calls if isinstance(item, tuple))
    assert window[1] == (
        "http://127.0.0.1:54321"
        if bundled
        else None
        if "--demo" in arguments
        else "http://127.0.0.1:5555"
    )
    assert window[3] == paths.default_output_directory()
    assert ("service stopped" in calls) == bundled


def test_data_paths_and_metadata_belong_to_training_repository(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert paths.repository_root() == root
    assert paths.default_output_directory() == root / "data/experiments"
    metadata = app.software_metadata()
    assert metadata["software"] == "bibazu-autotune-training"
    assert metadata["version"] == version("bibazu-autotune-training")
    assert metadata["autotune_version"] == version("bibazu-autotune")
    assert "git_commit" in metadata and "autotune_git_commit" in metadata
    monkeypatch.setattr(paths, "repository_root", lambda: None)
    assert (
        paths.default_output_directory()
        == Path.home() / "Documents/BiBaZu_AutoTune_Training/experiments"
    )


def test_training_launcher_works_outside_repository(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows launcher")
    launcher = Path(__file__).resolve().parents[1] / "Start.ps1"
    if not launcher.is_file():
        pytest.skip("Only applicable to the editable workspace installation")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--vision-url" in result.stdout
