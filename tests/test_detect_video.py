import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import detect_video as dv  # noqa: E402

CLASS_NAMES = ["box", "bicycle", "stroller"]


def test_should_infer_on_first_frame():
    assert dv.should_infer(0, 90) is True


def test_should_infer_every_stride_frames():
    assert dv.should_infer(90, 90) is True
    assert dv.should_infer(180, 90) is True


def test_should_not_infer_between_strides():
    assert dv.should_infer(1, 90) is False
    assert dv.should_infer(89, 90) is False
    assert dv.should_infer(91, 90) is False


def test_should_infer_rejects_non_positive_stride():
    with pytest.raises(ValueError):
        dv.should_infer(0, 0)


def test_class_color_is_stable_and_distinct():
    colors = {dv.class_color(i) for i in range(3)}
    assert len(colors) == 3
    assert dv.class_color(0) == dv.class_color(0)
    for c in colors:
        assert len(c) == 3
        assert all(0 <= v <= 255 for v in c)


class _FakeBoxes:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        for xyxy, cls_id, conf in self._rows:
            yield type(
                "B",
                (),
                {
                    "xyxy": np.array([xyxy], dtype=float),
                    "cls": np.array([cls_id], dtype=float),
                    "conf": np.array([conf], dtype=float),
                },
            )()


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)


def test_parse_results_converts_boxes():
    result = _FakeResult([([10.4, 20.6, 100.2, 200.9], 2, 0.87)])
    dets = dv.parse_results(result, conf_threshold=0.25)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (10, 21, 100, 201)
    assert d.cls_id == 2
    assert d.conf == pytest.approx(0.87)


def test_parse_results_filters_low_confidence():
    result = _FakeResult(
        [([0, 0, 10, 10], 0, 0.10), ([5, 5, 20, 20], 1, 0.90)]
    )
    dets = dv.parse_results(result, conf_threshold=0.25)
    assert [d.cls_id for d in dets] == [1]


def test_parse_results_handles_empty():
    assert dv.parse_results(_FakeResult([]), conf_threshold=0.25) == []


def test_parse_results_handles_none_boxes():
    assert dv.parse_results(type("R", (), {"boxes": None})(), 0.25) == []


def test_draw_detections_modifies_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = [dv.Detection(x1=10, y1=10, x2=100, y2=100, cls_id=0, conf=0.9)]
    out = dv.draw_detections(frame.copy(), dets, CLASS_NAMES)
    assert out.shape == frame.shape
    assert out.any(), "박스가 그려지지 않았습니다"


def test_draw_detections_with_empty_list_keeps_frame_unchanged():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = dv.draw_detections(frame.copy(), [], CLASS_NAMES)
    assert not out.any()


def test_draw_detections_clamps_out_of_bounds_boxes():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = [dv.Detection(x1=-50, y1=-50, x2=999, y2=999, cls_id=1, conf=0.5)]
    out = dv.draw_detections(frame.copy(), dets, CLASS_NAMES)
    assert out.shape == frame.shape


def _write_test_video(path: Path, frames: int = 200, size=(320, 240)) -> Path:
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size
    )
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_run_infers_only_every_stride_frames(tmp_path, monkeypatch):
    video = _write_test_video(tmp_path / "clip.mp4", frames=200)
    calls = {"n": 0}

    class _FakeModel:
        names = {0: "box", 1: "bicycle", 2: "stroller"}

        def predict(self, *a, **k):
            calls["n"] += 1
            return [_FakeResult([([10, 10, 50, 50], 0, 0.9)])]

    monkeypatch.setattr(dv, "load_model", lambda weights: _FakeModel())
    stats = dv.run(
        source=video,
        weights=Path("unused.pt"),
        stride=90,
        conf=0.25,
        save_path=tmp_path / "out.mp4",
        show=False,
    )
    assert stats["frames"] == 200
    assert stats["inferences"] == calls["n"] == 3  # frame 0, 90, 180
    assert (tmp_path / "out.mp4").exists()


def test_run_raises_on_missing_video(tmp_path):
    with pytest.raises(SystemExit):
        dv.run(
            source=tmp_path / "nope.mp4",
            weights=Path("unused.pt"),
            stride=90,
            conf=0.25,
            save_path=None,
            show=False,
        )
