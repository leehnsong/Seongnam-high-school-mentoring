import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train as tr  # noqa: E402


def test_resolve_device_returns_mps():
    assert tr.resolve_device("mps") == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(tr, "_mps_available", lambda: False)
    assert tr.resolve_device("mps") == "cpu"


def test_resolve_device_respects_explicit_cpu():
    assert tr.resolve_device("cpu") == "cpu"


def test_build_train_kwargs_contains_required_settings(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 3\n")
    kwargs = tr.build_train_kwargs(
        data_yaml, epochs=80, batch=16, imgsz=640, device="mps", run_name="loadobj"
    )
    assert kwargs["data"] == str(data_yaml)
    assert kwargs["epochs"] == 80
    assert kwargs["batch"] == 16
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "mps"
    assert kwargs["name"] == "loadobj"
    assert kwargs["patience"] == 15


def test_export_best_weights_copies_file(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    dest = tmp_path / "models" / "best.pt"
    out = tr.export_best_weights(run_dir, dest)
    assert out == dest
    assert dest.read_bytes() == b"weights"


def test_export_best_weights_raises_when_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        tr.export_best_weights(run_dir, tmp_path / "models" / "best.pt")


def test_mps_fallback_env_is_set():
    import os

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
