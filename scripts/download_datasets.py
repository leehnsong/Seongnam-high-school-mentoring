#!/usr/bin/env python3
"""Roboflow Universe에서 원본 데이터셋들을 datasets/raw/ 아래로 내려받는다."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
EXPORT_FORMAT = "yolov8"  # YOLO11과 동일한 디렉토리/라벨 포맷


def load_config(path: Path) -> dict:
    """데이터셋 설정 YAML을 읽어 dict로 반환한다."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    """.env 또는 환경변수에서 Roboflow API 키를 읽는다. 없으면 안내 후 종료."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ROBOFLOW_API_KEY를 찾을 수 없습니다.\n"
            "프로젝트 루트에 .env 파일을 만들고 아래 한 줄을 넣어주세요:\n"
            "  ROBOFLOW_API_KEY=발급받은키\n"
            "키는 https://app.roboflow.com/settings/api 에서 확인할 수 있습니다."
        )
    return key


def pick_version_number(available: list[int], explicit: int | None) -> int:
    """사용할 데이터셋 버전 번호를 고른다. explicit이 없으면 가장 높은 버전."""
    if not available:
        raise ValueError("사용 가능한 버전이 없습니다.")
    if explicit is not None:
        if explicit not in available:
            raise ValueError(
                f"버전 {explicit}이 없습니다. 사용 가능: {sorted(available)}"
            )
        return explicit
    return max(available)


def _version_numbers(project) -> list[int]:
    """roboflow project 객체에서 버전 번호 정수 목록을 뽑는다."""
    numbers = []
    for v in project.versions():
        raw = str(getattr(v, "version", "")).rstrip("/").split("/")[-1]
        if raw.isdigit():
            numbers.append(int(raw))
    return numbers


def download_source(rf, source: dict, dest_root: Path) -> Path:
    """소스 하나를 dest_root/<name>/ 으로 내려받고 그 경로를 반환한다."""
    name = source["name"]
    dest = Path(dest_root) / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    project = rf.workspace(source["workspace"]).project(source["project"])
    number = pick_version_number(_version_numbers(project), source.get("version"))
    print(f"  [{name}] {source['workspace']}/{source['project']} v{number} 다운로드 중...")
    project.version(number).download(EXPORT_FORMAT, location=str(dest))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Roboflow 데이터셋 다운로드")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()

    from roboflow import Roboflow

    cfg = load_config(args.config)
    rf = Roboflow(api_key=get_api_key())

    ok, failed = [], []
    for source in cfg["sources"]:
        try:
            download_source(rf, source, args.out)
            ok.append(source["name"])
        except Exception as exc:  # noqa: BLE001 — 개별 실패는 건너뛰고 계속
            print(f"  [{source['name']}] 실패: {type(exc).__name__}: {exc}")
            failed.append(source["name"])

    print(f"\n성공 {len(ok)}개: {', '.join(ok) if ok else '-'}")
    if failed:
        print(f"실패 {len(failed)}개: {', '.join(failed)}")
        print("→ configs/datasets.yaml 에서 해당 항목의 workspace/project를 확인하거나")
        print("  https://universe.roboflow.com 에서 대체 데이터셋을 찾아 교체하세요.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
