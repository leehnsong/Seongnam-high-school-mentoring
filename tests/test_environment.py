"""환경이 올바르게 준비되었는지 검증하는 스모크 테스트."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_required_packages_importable():
    import cv2  # noqa: F401
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    import yaml  # noqa: F401


def test_mps_is_available():
    import torch

    assert torch.backends.mps.is_available(), (
        "MPS를 사용할 수 없습니다. Apple Silicon Mac + PyTorch 2.x 인지 확인하세요."
    )


def test_mps_tensor_roundtrip():
    import torch

    x = torch.ones(4, 4, device="mps")
    assert float((x * 2).sum().cpu()) == 32.0


def test_project_directories_exist():
    for rel in [
        "configs",
        "scripts",
        "tests",
        "datasets/raw",
        "datasets/merged",
        "models",
        "outputs",
    ]:
        assert (PROJECT_ROOT / rel).is_dir(), f"디렉토리 없음: {rel}"
