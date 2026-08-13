#!/usr/bin/env python3
"""YOLO11s를 3클래스 적재물 데이터셋으로 fine-tuning한다 (Apple MPS)."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# MPS가 지원하지 않는 연산은 CPU로 자동 폴백 (import torch 이전에 설정해야 함)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "merged" / "data.yaml"
DEFAULT_BASE_MODEL = "yolo11s.pt"
DEFAULT_DEST = PROJECT_ROOT / "models" / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"


def _mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_available())


def resolve_device(requested: str) -> str:
    """요청한 device를 검증한다. mps를 못 쓰면 cpu로 폴백."""
    if requested == "mps" and not _mps_available():
        print("경고: MPS를 사용할 수 없어 CPU로 학습합니다 (매우 느립니다).")
        return "cpu"
    return requested


def build_train_kwargs(
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    run_name: str,
) -> dict:
    """Ultralytics train()에 넘길 인자를 만든다."""
    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": 4,
        "patience": 15,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": True,
        "seed": 0,
        "plots": True,
    }


def export_best_weights(run_dir: Path, dest: Path) -> Path:
    """학습 결과 best.pt를 models/best.pt로 복사한다."""
    src = Path(run_dir) / "weights" / "best.pt"
    if not src.exists():
        raise FileNotFoundError(f"학습 가중치를 찾을 수 없습니다: {src}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="적재물 detection 모델 학습")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="loadobj")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"데이터셋이 없습니다: {args.data}\n"
            "먼저 scripts/download_datasets.py 와 scripts/merge_datasets.py 를 실행하세요."
        )

    from ultralytics import YOLO

    device = resolve_device(args.device)
    kwargs = build_train_kwargs(
        args.data, args.epochs, args.batch, args.imgsz, device, args.name
    )
    print(f"학습 시작 — device={device}, batch={args.batch}, epochs={args.epochs}")

    model = YOLO(args.model)
    results = model.train(**kwargs)

    run_dir = Path(results.save_dir)
    dest = export_best_weights(run_dir, args.dest)
    print(f"\n학습 완료. 가중치 저장: {dest}")
    print(f"학습 로그/그래프: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
