import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import download_datasets as dl  # noqa: E402


def test_load_config_returns_classes_and_sources():
    cfg = dl.load_config(PROJECT_ROOT / "configs" / "datasets.yaml")
    assert cfg["classes"] == ["box", "bicycle", "stroller"]
    assert len(cfg["sources"]) >= 1


def test_every_source_has_required_fields():
    cfg = dl.load_config(PROJECT_ROOT / "configs" / "datasets.yaml")
    names = set()
    for src in cfg["sources"]:
        assert src["name"] not in names, f"중복된 source name: {src['name']}"
        names.add(src["name"])
        assert src["workspace"]
        assert src["project"]
        has_target = "target" in src
        has_map = "class_map" in src
        assert has_target != has_map, (
            f"{src['name']}: target 또는 class_map 중 정확히 하나만 있어야 합니다"
        )


def test_source_targets_reference_known_classes():
    cfg = dl.load_config(PROJECT_ROOT / "configs" / "datasets.yaml")
    known = set(cfg["classes"])
    for src in cfg["sources"]:
        if "target" in src:
            assert src["target"] in known
        else:
            for unified in src["class_map"].values():
                assert unified in known


def test_pick_version_number_prefers_explicit():
    assert dl.pick_version_number([1, 2, 5], explicit=2) == 2


def test_pick_version_number_defaults_to_highest():
    assert dl.pick_version_number([1, 7, 3], explicit=None) == 7


def test_pick_version_number_rejects_missing_explicit():
    with pytest.raises(ValueError):
        dl.pick_version_number([1, 2], explicit=9)


def test_pick_version_number_rejects_empty():
    with pytest.raises(ValueError):
        dl.pick_version_number([], explicit=None)


def test_get_api_key_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        dl.get_api_key()


def test_get_api_key_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "dummy-key")
    assert dl.get_api_key() == "dummy-key"
