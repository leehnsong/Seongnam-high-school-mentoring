#!/usr/bin/env python3
"""녹화 영상에서 적재물을 90프레임 간격으로 탐지해 박스를 그려 보여준다."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_STRIDE = 90

# 클래스별 색상 (BGR) — 0=box, 1=bicycle, 2=stroller
CLASS_COLORS = [(0, 165, 255), (0, 255, 0), (255, 80, 80)]


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


def should_infer(frame_index: int, stride: int) -> bool:
    """이 프레임에서 추론해야 하는지 판단한다."""
    if stride <= 0:
        raise ValueError("stride는 1 이상이어야 합니다.")
    return frame_index % stride == 0


def class_color(cls_id: int) -> tuple[int, int, int]:
    """클래스 ID에 대응하는 고정 색상(BGR)을 반환한다."""
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


def parse_results(result: Any, conf_threshold: float) -> list[Detection]:
    """Ultralytics 결과 객체를 Detection 리스트로 변환한다."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    detections: list[Detection] = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            Detection(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                cls_id=int(box.cls[0]),
                conf=conf,
            )
        )
    return detections


def draw_detections(frame, detections: list[Detection], class_names: list[str]):
    """프레임 위에 박스와 라벨을 그린다."""
    height, width = frame.shape[:2]
    for det in detections:
        x1 = max(0, min(det.x1, width - 1))
        y1 = max(0, min(det.y1, height - 1))
        x2 = max(0, min(det.x2, width - 1))
        y2 = max(0, min(det.y2, height - 1))
        color = class_color(det.cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        name = class_names[det.cls_id] if det.cls_id < len(class_names) else "?"
        label = f"{name} {det.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - th - baseline - 2)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 2, top + th + baseline + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 1, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def load_model(weights: Path):
    """학습된 YOLO 모델을 로드한다."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise SystemExit(
            f"모델 가중치가 없습니다: {weights}\n먼저 scripts/train.py 를 실행하세요."
        )
    return YOLO(str(weights))


def run(
    source: Path,
    weights: Path,
    stride: int,
    conf: float,
    save_path: Path | None,
    show: bool,
) -> dict:
    """영상을 순회하며 stride 프레임마다 추론하고, 사이 프레임은 직전 결과를 재사용한다."""
    source = Path(source)
    if not source.exists():
        raise SystemExit(f"영상 파일을 찾을 수 없습니다: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다 (코덱 문제일 수 있습니다): {source}")

    model = load_model(weights)
    names = model.names
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    last_detections: list[Detection] = []
    frame_index = 0
    inferences = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if should_infer(frame_index, stride):
                results = model.predict(frame, conf=conf, verbose=False)
                last_detections = parse_results(results[0], conf) if results else []
                inferences += 1
                print(
                    f"  frame {frame_index}: {len(last_detections)}개 탐지 "
                    f"({', '.join(class_names[d.cls_id] for d in last_detections) or '-'})"
                )

            # 추론하지 않은 프레임에도 직전 탐지 결과를 그대로 그린다.
            drawn = draw_detections(frame, last_detections, class_names)

            if writer is not None:
                writer.write(drawn)
            if show:
                cv2.imshow("loading detection (q: 종료)", drawn)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return {"frames": frame_index, "inferences": inferences}


def main() -> int:
    parser = argparse.ArgumentParser(description="영상 적재물 탐지")
    parser.add_argument("--source", type=Path, required=True, help="입력 영상 파일")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-show", action="store_true", help="화면 표시 끄기")
    parser.add_argument("--no-save", action="store_true", help="결과 영상 저장 안 함")
    args = parser.parse_args()

    save_path = None
    if not args.no_save:
        save_path = DEFAULT_OUTPUT_DIR / f"{args.source.stem}_detected.mp4"

    stats = run(
        source=args.source,
        weights=args.weights,
        stride=args.stride,
        conf=args.conf,
        save_path=save_path,
        show=not args.no_show,
    )
    print(f"\n총 {stats['frames']}프레임 처리, {stats['inferences']}회 추론")
    if save_path is not None:
        print(f"결과 영상: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
