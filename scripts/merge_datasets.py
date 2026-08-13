#!/usr/bin/env python3
"""datasets/raw/의 여러 데이터셋을 3클래스 단일 데이터셋으로 병합한다."""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "merged"

# Roboflow는 valid/ 로 내보내지만 우리는 val/ 로 통일한다.
SPLIT_ALIASES = {"train": ["train"], "val": ["valid", "val"]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_source_class_names(data_yaml: Path) -> list[str]:
    """원본 데이터셋 data.yaml에서 클래스 이름을 인덱스 순서대로 읽는다."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def build_id_mapping(
    source_names: list[str], spec: dict, unified: list[str]
) -> dict[int, int]:
    """원본 클래스 ID → 통합 클래스 ID 매핑을 만든다. 매핑에 없는 ID는 버려진다."""
    if "target" in spec:
        target = spec["target"]
        if target not in unified:
            raise ValueError(f"알 수 없는 통합 클래스: {target}")
        return {i: unified.index(target) for i in range(len(source_names))}

    mapping: dict[int, int] = {}
    for src_name, unified_name in spec["class_map"].items():
        if unified_name not in unified:
            raise ValueError(f"알 수 없는 통합 클래스: {unified_name}")
        if src_name not in source_names:
            raise ValueError(
                f"원본에 없는 클래스명: {src_name} (사용 가능: {source_names})"
            )
        mapping[source_names.index(src_name)] = unified.index(unified_name)
    return mapping


def remap_label_text(text: str, mapping: dict[int, int]) -> str:
    """YOLO 라벨 텍스트의 클래스 ID를 재매핑한다. 형식 오류·미매핑 줄은 버린다."""
    out_lines = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
        except ValueError:
            continue
        if src_id not in mapping:
            continue
        out_lines.append(" ".join([str(mapping[src_id]), *parts[1:]]))
    return "".join(f"{line}\n" for line in out_lines)


def merge_split(
    src_dir: Path, dst_dir: Path, prefix: str, mapping: dict[int, int]
) -> dict[str, int]:
    """src_dir(images/, labels/)의 한 split을 dst_dir로 복사하며 라벨을 재매핑한다."""
    src_images = Path(src_dir) / "images"
    src_labels = Path(src_dir) / "labels"
    dst_images = Path(dst_dir) / "images"
    dst_labels = Path(dst_dir) / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter({"copied": 0, "skipped_no_label": 0, "skipped_empty_after_remap": 0})
    if not src_images.is_dir():
        return dict(stats)

    for image_path in sorted(src_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["skipped_no_label"] += 1
            continue
        remapped = remap_label_text(
            label_path.read_text(encoding="utf-8", errors="ignore"), mapping
        )
        if not remapped:
            stats["skipped_empty_after_remap"] += 1
            continue
        new_stem = f"{prefix}__{image_path.stem}"
        shutil.copy2(image_path, dst_images / f"{new_stem}{image_path.suffix}")
        (dst_labels / f"{new_stem}.txt").write_text(remapped, encoding="utf-8")
        stats["copied"] += 1
    return dict(stats)


def write_data_yaml(root: Path, classes: list[str]) -> Path:
    """학습용 data.yaml을 생성하고 경로를 반환한다."""
    root = Path(root).resolve()
    out = root / "data.yaml"
    out.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "nc": len(classes),
                "names": list(classes),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return out


def _find_split_dir(source_root: Path, split: str) -> Path | None:
    for alias in SPLIT_ALIASES[split]:
        candidate = source_root / alias
        if (candidate / "images").is_dir():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="데이터셋 병합")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    unified = cfg["classes"]

    if args.out.exists():
        shutil.rmtree(args.out)

    totals = Counter()
    per_class = Counter()
    for spec in cfg["sources"]:
        source_root = Path(args.raw) / spec["name"]
        data_yaml = source_root / "data.yaml"
        if not data_yaml.exists():
            print(f"  [{spec['name']}] 건너뜀 — 다운로드되지 않음")
            continue
        names = read_source_class_names(data_yaml)
        mapping = build_id_mapping(names, spec, unified)

        for split in ("train", "val"):
            split_dir = _find_split_dir(source_root, split)
            if split_dir is None:
                continue
            stats = merge_split(
                split_dir, Path(args.out) / split, spec["name"], mapping
            )
            totals.update({f"{split}_{k}": v for k, v in stats.items()})
            print(f"  [{spec['name']}/{split}] {stats}")

        for unified_id in set(mapping.values()):
            per_class[unified[unified_id]] += 1

    # YOLO 표준 레이아웃(images/train, labels/train)으로 재배치
    final = Path(args.out)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src = final / split / kind
            dst = final / kind / split
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                dst.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final / split, ignore_errors=True)

    write_data_yaml(final, unified)
    print(f"\n합계: {dict(totals)}")
    print(f"클래스별 기여 데이터셋 수: {dict(per_class)}")
    print(f"data.yaml 생성: {final / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
