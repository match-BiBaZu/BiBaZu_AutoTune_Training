"""Training runner over the real shared HTTP service, with an entirely in-memory PLC."""

import copy
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from bibazu_autotune import pressure_service
from bibazu_autotune.clients import ApiFailure
from bibazu_autotune.profiles import empty_parameters
from test_experiments import plan_assets as plan_assets

from bibazu_autotune_training.app import PressureClient
from bibazu_autotune_training.demo import DemoVision
from bibazu_autotune_training.runner import ExperimentRunner


class MemoryBackend:
    def __init__(self):
        self.parameters = empty_parameters()
        self.moving = self.closed = False
        self.writes = []

    def connect(self):
        pass

    def snapshot(self):
        return {"parameters": copy.deepcopy(self.parameters), "conveyor": {"moving": self.moving}}

    def machine_metadata(self):
        return {"simulated": True}

    def apply(self, parameters, guard):
        guard()
        self.parameters = copy.deepcopy(parameters)
        self.writes.append("parameters")
        return self.snapshot()

    def start_conveyor(self, guard):
        guard()
        self.moving = True
        self.writes.append("start")
        return self.snapshot()

    def stop_conveyor(self):
        self.moving = False
        self.writes.append("stop")
        return self.snapshot()

    def close(self):
        self.closed = True


def test_training_uses_shared_plc_service_and_session_exclusion(plan_assets, tmp_path, monkeypatch):
    backend = MemoryBackend()
    lease = SimpleNamespace(closed=False)
    lease.close = lambda: setattr(lease, "closed", True)
    service = pressure_service.PressureService(
        backend, mutex_factory=lambda: lease, pressure_test_dir=tmp_path / "pressure-tests"
    )
    monkeypatch.setattr(pressure_service, "configured_service", lambda _: service)
    running, server = pressure_service.start_bundled(SimpleNamespace())
    assert running is service and not backend.writes
    url = f"http://127.0.0.1:{server.port}"
    client, competitor = PressureClient(url), PressureClient(url)
    client.simulated = True  # Only this in-memory backend may accept the synthetic camera.
    vision = DemoVision(backend)
    runner = ExperimentRunner(client, vision)

    def until(state):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            runner.tick()
            if runner.state == state:
                return
            time.sleep(0.01)
        raise AssertionError(f"Expected {state}, got {runner.state}")

    try:
        runner.connect()
        until("ready")
        assert not backend.writes
        with pytest.raises(ApiFailure, match="409"):
            competitor.acquire()
        plan, assets = plan_assets
        runner.create(tmp_path / "campaigns", replace(plan, settling_seconds=0), assets, {})
        until("gate")
        runner.confirm()
        until("waiting")
        assert not backend.moving and "start" not in backend.writes
        runner.conveyor(True)
        until("gate")
        records = runner.store.trials()
        assert len(records) == 2
        assert all(record["resolution"] == "accepted" for record in records)
        assert backend.moving
        assert all(not array["enabled"] for array in backend.parameters["arrays"])
        assert not (tmp_path / "pressure-tests").exists()
    finally:
        runner.close()
        competitor.close()
        vision.close()
        server.stop()
        service.stop()
    assert backend.closed and not backend.moving and lease.closed
    assert not service.thread.is_alive()
