# YOLO 적재물 Detection 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 녹화된 영상에서 적재물(box / bicycle / stroller)의 종류와 위치를 90프레임마다 탐지해 화면에 박스로 표시하는 프로그램을, Mac MPS 환경에서 YOLO11s fine-tuning으로 구현한다.

**Architecture:** 4개의 독립 스크립트 파이프라인 — ① Roboflow에서 원본 데이터셋 다운로드 → ② 클래스 ID를 재매핑해 3클래스 단일 데이터셋으로 병합 → ③ YOLO11s fine-tuning(device=mps) → ④ 영상에서 90프레임 간격 추론 후 박스 오버레이. 각 스크립트는 파일시스템(`datasets/`, `models/`)을 통해서만 연결되므로 따로 실행·테스트할 수 있다.

**Tech Stack:** Python 3.11 (conda), Ultralytics YOLO11, PyTorch(MPS), OpenCV, Roboflow SDK, PyYAML, python-dotenv, pytest

## Global Constraints

- conda 환경 이름: `yolo-load`, Python 버전: `3.11`
- 통합 클래스는 **이 순서로 고정**: `0=box`, `1=bicycle`, `2=stroller`
- 학습·추론 device는 `mps`. 모든 실행 스크립트는 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정한다.
- 학습 기본 하이퍼파라미터: `imgsz=640`, `batch=16`, `epochs=80`, `patience=15`
- 추론 간격(stride): **90 프레임**. 추론하지 않는 프레임은 직전 탐지 결과를 그대로 그린다.
- Roboflow API 키는 **오직 `.env` 파일**에서만 읽는다. 코드·설정파일·커밋 메시지·로그에 키 값을 절대 넣지 않는다.
- 모든 경로는 `PROJECT_ROOT = Path(__file__).resolve().parent.parent` 기준 절대경로로 계산한다.
- 테스트는 `pytest`, 프로젝트 루트에서 `pytest tests/ -v`로 실행한다.

---

### Task 1: conda 환경 + 프로젝트 스캐폴드 + MPS 검증

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `README.md`
- Create: `tests/test_environment.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: conda 환경 `yolo-load`, 설치된 `ultralytics`/`opencv-python`/`roboflow`/`pyyaml`/`python-dotenv`/`pytest`, 그리고 `datasets/raw`, `datasets/merged`, `models`, `outputs`, `configs`, `scripts`, `tests` 디렉토리

- [ ] **Step 1: `environment.yml` 작성**

```yaml
name: yolo-load
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - -r requirements.txt
```

- [ ] **Step 2: `requirements.txt` 작성**

```
ultralytics>=8.3.0
torch>=2.4.0
torchvision>=0.19.0
opencv-python>=4.10.0
roboflow>=1.1.40
pyyaml>=6.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 3: conda 환경 생성**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring
conda env create -f environment.yml
```
Expected: 마지막에 `done` 과 `# To activate this environment, use ... conda activate yolo-load` 출력.

이후 모든 명령은 이 환경에서 실행한다. 각 Run 블록 앞에 `conda run -n yolo-load` 를 붙이거나, 셸에서 `conda activate yolo-load` 후 실행한다.

- [ ] **Step 4: 디렉토리 스캐폴드 생성**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring
mkdir -p configs scripts tests datasets/raw datasets/merged models outputs
touch datasets/.gitkeep models/.gitkeep outputs/.gitkeep
```

- [ ] **Step 5: `.gitignore` 갱신**

`.gitignore` 전체를 아래 내용으로 교체한다:

```
# secrets
.env

# python
__pycache__/
*.pyc
.pytest_cache/
.conda/

# data & model artifacts (대용량)
datasets/raw/
datasets/merged/
models/*.pt
outputs/*
runs/

# keep dir placeholders (부모 디렉토리가 통째로 무시되면 재포함이 불가하므로 outputs는 outputs/* 로 지정)
!datasets/.gitkeep
!models/.gitkeep
!outputs/.gitkeep
```

- [ ] **Step 6: `.env.example` 작성**

```
# Roboflow Private API Key — https://app.roboflow.com/settings/api 에서 복사
# 이 파일을 .env 로 복사한 뒤 실제 키를 넣으세요. .env 는 git에 커밋되지 않습니다.
ROBOFLOW_API_KEY=여기에_발급받은_키를_붙여넣으세요
```

- [ ] **Step 7: 실패하는 환경 테스트 작성**

`tests/test_environment.py`:

```python
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
```

- [ ] **Step 8: 테스트 실행 — 통과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_environment.py -v
```
Expected: 4 passed.

만약 `test_mps_is_available`가 실패하면 PyTorch가 CPU 빌드로 설치된 것이다. `conda run -n yolo-load pip install --force-reinstall torch torchvision` 후 재실행한다.

- [ ] **Step 9: `README.md` 작성**

````markdown
# 적재물 Detection (YOLO11 + MPS)

녹화된 영상에서 **박스 / 자전거 / 유모차** 를 탐지해 화면에 표시합니다.

| id | class | 정의 |
|----|-------|------|
| 0 | `box` | 박스처럼 생긴 것 전부 |
| 1 | `bicycle` | 일반 자전거만 (오토바이·킥보드 제외) |
| 2 | `stroller` | 아기 유모차만 |

## 설치

```bash
conda env create -f environment.yml
conda activate yolo-load
cp .env.example .env   # 그리고 .env 안에 Roboflow API 키를 넣으세요
```

## 실행 순서

```bash
python scripts/download_datasets.py     # ① 원본 데이터셋 다운로드
python scripts/merge_datasets.py        # ② 3클래스로 병합
python scripts/train.py                 # ③ 학습 (MPS)
python scripts/detect_video.py --source 영상.mp4   # ④ 영상 추론
```

## 알려진 한계

여러 개의 단일 클래스 데이터셋을 병합하기 때문에, 예를 들어 박스 데이터셋 이미지에
자전거가 찍혀 있어도 라벨이 없습니다. 이런 **미라벨 인스턴스**는 학습 시 약한 오답 신호가
됩니다. 클래스가 함께 등장하는 이미지가 많다고 판단되면 해당 이미지를 제외하거나
직접 라벨을 보강하세요.
````

- [ ] **Step 10: 커밋**

```bash
cd /Users/leehnsong/Desktop/mentoring
git add environment.yml requirements.txt .env.example README.md .gitignore tests/test_environment.py datasets/.gitkeep models/.gitkeep outputs/.gitkeep
git commit -m "chore: conda 환경 및 프로젝트 스캐폴드 추가"
```

---

### Task 2: 데이터셋 설정 + 다운로드 스크립트

**Files:**
- Create: `configs/datasets.yaml`
- Create: `scripts/download_datasets.py`
- Create: `tests/test_download_datasets.py`

**Interfaces:**
- Consumes: Task 1의 conda 환경, `.env`의 `ROBOFLOW_API_KEY`
- Produces:
  - `configs/datasets.yaml` — `{"classes": list[str], "sources": list[dict]}`
  - `scripts/download_datasets.py` 안의 함수:
    - `load_config(path: Path) -> dict`
    - `get_api_key() -> str`
    - `pick_version_number(available: list[int], explicit: int | None) -> int`
    - `download_source(rf, source: dict, dest_root: Path) -> Path`
  - 파일시스템: `datasets/raw/<source_name>/` (각각 `train/`, `valid/`, `data.yaml` 포함)

- [ ] **Step 1: `configs/datasets.yaml` 작성**

`target`는 "이 데이터셋의 모든 클래스를 이 통합 클래스로 취급", `class_map`은 "소스 클래스명 → 통합 클래스명, 목록에 없는 클래스는 버림"을 뜻한다. 둘 중 하나만 쓴다.

```yaml
# 통합 클래스 — 순서가 곧 클래스 ID (0, 1, 2). 절대 순서를 바꾸지 말 것.
classes:
  - box
  - bicycle
  - stroller

# Roboflow Universe 원본 데이터셋 목록.
# version을 생략하면 사용 가능한 가장 높은 버전을 자동 선택한다.
sources:
  - name: box_cardboard_main
    workspace: harshuu
    project: cardboard-box-detection-kh0qu
    target: box

  - name: box_open_closed
    workspace: cardbox-damage-detection
    project: custom-dataset-for-closed-and-open-cardboard-box
    target: box

  - name: mixed_stroller_bicycle
    workspace: thales-a5kye
    project: stroller_final
    class_map:
      stroller: stroller
      bicycle: bicycle

  - name: stroller_ultimate
    workspace: furnitureselectronics
    project: ultimite_strollers_detection
    target: stroller

  - name: bicycle_training_data
    workspace: yolov5-sbzvs
    project: bicycle-training-data
    target: bicycle

  - name: bicycle_bike_identification
    workspace: testing-qu1kc
    project: yolo-bike-identification-project
    target: bicycle
```

> **주의:** 위 workspace/project 슬러그는 Roboflow Universe 검색 결과에서 가져온 것이며,
> 데이터셋이 비공개로 바뀌거나 삭제되었을 수 있다. Step 7에서 실제로 다운로드해 검증하고,
> 실패한 항목은 Universe에서 대체 데이터셋을 찾아 슬러그를 교체한다.

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_download_datasets.py`:

```python
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
```

- [ ] **Step 3: 테스트 실행 — 실패 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_download_datasets.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'download_datasets'`

- [ ] **Step 4: `scripts/download_datasets.py` 구현**

```python
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
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_download_datasets.py -v
```
Expected: 9 passed.

- [ ] **Step 6: 사용자가 `.env` 생성했는지 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && test -f .env && echo "ENV_FILE_OK" || echo "ENV_FILE_MISSING"
```
Expected: `ENV_FILE_OK`

`ENV_FILE_MISSING`이면 사용자에게 다음을 요청하고 멈춘다:
> `.env` 파일이 필요합니다. 터미널에서 `cp .env.example .env` 후 `.env` 안의 값을 발급받으신 Roboflow 키로 바꿔주세요. (키는 채팅에 붙여넣지 마세요.)

- [ ] **Step 7: 실제 다운로드 실행 및 결과 검증**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/download_datasets.py
```
Expected: 각 소스에 대해 `다운로드 중...` 로그가 뜨고 마지막에 `성공 N개` 출력.

이어서 실제로 받아진 내용을 확인한다:
```bash
cd /Users/leehnsong/Desktop/mentoring && for d in datasets/raw/*/; do echo "== $d"; ls "$d"; echo "-- classes:"; grep -A2 '^names' "$d/data.yaml" 2>/dev/null | head -5; echo "-- train images: $(ls "$d"train/images 2>/dev/null | wc -l)"; done
```
Expected: 각 디렉토리에 `train/`, `valid/`, `data.yaml`이 있고 train 이미지 수가 0보다 크다.

**실패한 소스 처리:** 실패 항목이 있으면 https://universe.roboflow.com 에서 해당 클래스로 검색해 공개(Public) 데이터셋을 찾고, URL `universe.roboflow.com/<workspace>/<project>` 의 두 슬러그를 `configs/datasets.yaml`에 반영한 뒤 이 Step을 다시 실행한다. **클래스별로 최소 1개 소스는 반드시 성공해야 다음 태스크로 넘어간다.**

- [ ] **Step 8: 클래스별 최소 커버리지 검증**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python -c "
import sys, pathlib
sys.path.insert(0, 'scripts')
import yaml, download_datasets as dl
cfg = dl.load_config(pathlib.Path('configs/datasets.yaml'))
raw = pathlib.Path('datasets/raw')
covered = set()
for s in cfg['sources']:
    if not (raw / s['name'] / 'data.yaml').exists():
        continue
    covered |= {s['target']} if 'target' in s else set(s['class_map'].values())
missing = set(cfg['classes']) - covered
print('covered:', sorted(covered))
assert not missing, f'다운로드된 데이터가 없는 클래스: {sorted(missing)}'
print('COVERAGE_OK')
"
```
Expected: `COVERAGE_OK`

- [ ] **Step 9: 커밋**

```bash
cd /Users/leehnsong/Desktop/mentoring
git add configs/datasets.yaml scripts/download_datasets.py tests/test_download_datasets.py
git commit -m "feat: Roboflow 데이터셋 설정 및 다운로드 스크립트 추가"
```

---

### Task 3: 데이터셋 병합 (클래스 ID 재매핑)

**Files:**
- Create: `scripts/merge_datasets.py`
- Create: `tests/test_merge_datasets.py`

**Interfaces:**
- Consumes: `datasets/raw/<name>/` (Task 2), `configs/datasets.yaml`의 `classes`/`sources`
- Produces:
  - `scripts/merge_datasets.py` 안의 함수:
    - `read_source_class_names(data_yaml: Path) -> list[str]`
    - `build_id_mapping(source_names: list[str], spec: dict, unified: list[str]) -> dict[int, int]`
    - `remap_label_text(text: str, mapping: dict[int, int]) -> str`
    - `merge_split(src_dir: Path, dst_dir: Path, prefix: str, mapping: dict[int, int]) -> dict[str, int]`
    - `write_data_yaml(root: Path, classes: list[str]) -> Path`
  - 파일시스템: `datasets/merged/images/{train,val}/`, `datasets/merged/labels/{train,val}/`, `datasets/merged/data.yaml`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_merge_datasets.py`:

```python
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import merge_datasets as md  # noqa: E402

UNIFIED = ["box", "bicycle", "stroller"]


def test_read_source_class_names(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"names": ["cardboard", "damaged"], "nc": 2}))
    assert md.read_source_class_names(p) == ["cardboard", "damaged"]


def test_read_source_class_names_handles_dict_form(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"names": {0: "cardboard", 1: "damaged"}, "nc": 2}))
    assert md.read_source_class_names(p) == ["cardboard", "damaged"]


def test_build_id_mapping_with_target_maps_every_class():
    mapping = md.build_id_mapping(["a", "b", "c"], {"target": "box"}, UNIFIED)
    assert mapping == {0: 0, 1: 0, 2: 0}


def test_build_id_mapping_with_class_map_drops_unlisted():
    spec = {"class_map": {"stroller": "stroller", "bicycle": "bicycle"}}
    names = ["human", "wheelchair", "suitcase", "stroller", "bicycle"]
    assert md.build_id_mapping(names, spec, UNIFIED) == {3: 2, 4: 1}


def test_build_id_mapping_rejects_unknown_unified_class():
    with pytest.raises(ValueError):
        md.build_id_mapping(["a"], {"target": "truck"}, UNIFIED)


def test_build_id_mapping_rejects_source_class_not_present():
    with pytest.raises(ValueError):
        md.build_id_mapping(["a", "b"], {"class_map": {"zzz": "box"}}, UNIFIED)


def test_remap_label_text_rewrites_class_ids():
    text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"
    assert md.remap_label_text(text, {0: 2, 1: 1}) == (
        "2 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"
    )


def test_remap_label_text_drops_unmapped_classes():
    text = "0 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.05 0.05\n"
    assert md.remap_label_text(text, {0: 1}) == "1 0.5 0.5 0.2 0.2\n"


def test_remap_label_text_skips_malformed_lines():
    text = "0 0.5 0.5\nnot a label\n1 0.2 0.2 0.1 0.1\n"
    assert md.remap_label_text(text, {0: 0, 1: 1}) == "1 0.2 0.2 0.1 0.1\n"


def test_remap_label_text_returns_empty_for_no_valid_rows():
    assert md.remap_label_text("9 0.1 0.1 0.1 0.1\n", {0: 0}) == ""


def _make_split(root: Path, stem: str, label_text: str):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "labels" / f"{stem}.txt").write_text(label_text)


def test_merge_split_copies_and_prefixes(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "img1", "0 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="boxset", mapping={0: 0})
    assert stats["copied"] == 1
    assert (dst / "images" / "boxset__img1.jpg").exists()
    assert (dst / "labels" / "boxset__img1.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_merge_split_skips_image_without_label(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "images" / "lonely.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0})
    assert stats["copied"] == 0
    assert stats["skipped_no_label"] == 1


def test_merge_split_skips_when_all_boxes_dropped(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "img1", "5 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0})
    assert stats["copied"] == 0
    assert stats["skipped_empty_after_remap"] == 1


def test_write_data_yaml_has_three_classes(tmp_path):
    out = md.write_data_yaml(tmp_path, UNIFIED)
    data = yaml.safe_load(out.read_text())
    assert data["nc"] == 3
    assert data["names"] == UNIFIED
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert Path(data["path"]).is_absolute()
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_merge_datasets.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'merge_datasets'`

- [ ] **Step 3: `scripts/merge_datasets.py` 구현**

```python
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

    stats = Counter()
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_merge_datasets.py -v
```
Expected: 14 passed.

- [ ] **Step 5: 실제 병합 실행**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/merge_datasets.py
```
Expected: 소스별 통계가 출력되고 마지막에 `data.yaml 생성: .../datasets/merged/data.yaml`

- [ ] **Step 6: 병합 결과 검증**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python -c "
from pathlib import Path
from collections import Counter
import yaml
root = Path('datasets/merged')
data = yaml.safe_load((root / 'data.yaml').read_text())
assert data['nc'] == 3 and data['names'] == ['box','bicycle','stroller'], data
counts = Counter()
for split in ('train','val'):
    imgs = list((root / 'images' / split).glob('*'))
    lbls = list((root / 'labels' / split).glob('*.txt'))
    print(split, 'images:', len(imgs), 'labels:', len(lbls))
    assert len(imgs) == len(lbls), f'{split}: 이미지/라벨 개수 불일치'
    for p in lbls:
        for line in p.read_text().splitlines():
            counts[int(line.split()[0])] += 1
print('클래스별 박스 수:', {['box','bicycle','stroller'][k]: v for k, v in sorted(counts.items())})
assert set(counts) == {0,1,2}, f'세 클래스가 모두 있어야 합니다: {sorted(counts)}'
assert len(list((root/'images'/'train').glob('*'))) > 100, 'train 이미지가 너무 적습니다'
print('MERGE_OK')
"
```
Expected: `MERGE_OK` 와 함께 세 클래스 모두 0이 아닌 박스 수.

클래스 하나가 비어 있거나 train 이미지가 100장 미만이면 Task 2 Step 7로 돌아가 데이터셋을 더 추가한다.

- [ ] **Step 7: 커밋**

```bash
cd /Users/leehnsong/Desktop/mentoring
git add scripts/merge_datasets.py tests/test_merge_datasets.py
git commit -m "feat: 3클래스 데이터셋 병합 스크립트 추가"
```

---

### Task 4: YOLO11s 학습 (MPS)

**Files:**
- Create: `scripts/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `datasets/merged/data.yaml` (Task 3)
- Produces:
  - `scripts/train.py` 안의 함수:
    - `resolve_device(requested: str) -> str`
    - `build_train_kwargs(data_yaml: Path, epochs: int, batch: int, imgsz: int, device: str, run_name: str) -> dict`
    - `export_best_weights(run_dir: Path, dest: Path) -> Path`
  - 파일시스템: `models/best.pt` (학습된 3클래스 가중치)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_train.py`:

```python
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train as tr  # noqa: E402


def test_resolve_device_returns_mps_when_available():
    import torch

    if torch.backends.mps.is_available():
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_train.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: `scripts/train.py` 구현**

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_train.py -v
```
Expected: 7 passed.

- [ ] **Step 5: 1 epoch 스모크 학습으로 파이프라인 검증**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/train.py --epochs 1 --name smoke --dest models/smoke.pt
```
Expected: 학습이 끝까지 돌고 `학습 완료. 가중치 저장: .../models/smoke.pt` 출력. 에러 없이 완료되면 성공(정확도는 무의미).

`batch=16`에서 메모리 오류가 나면 `--batch 8`로 재시도한다. 반대로 여유가 많으면 본 학습에서 `--batch 32`를 써도 된다.

- [ ] **Step 6: 본 학습 실행**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/train.py --epochs 80 --batch 16 --name loadobj
```
Expected: 80 epoch(또는 early stop) 완료 후 `학습 완료. 가중치 저장: .../models/best.pt`

> 데이터 양에 따라 수십 분~수 시간이 걸린다. 백그라운드로 실행하고 진행 상황을 확인해도 된다.

- [ ] **Step 7: 학습 결과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python -c "
from ultralytics import YOLO
m = YOLO('models/best.pt')
print('names:', m.names)
assert list(m.names.values()) == ['box','bicycle','stroller'], m.names
print('MODEL_OK')
"
```
Expected: `names: {0: 'box', 1: 'bicycle', 2: 'stroller'}` 와 `MODEL_OK`

`runs/loadobj/results.png` 을 열어 mAP50 곡선이 상승했는지 확인한다. mAP50이 0.5 미만이면 데이터를 더 추가하거나 epoch를 늘린다.

- [ ] **Step 8: 커밋**

```bash
cd /Users/leehnsong/Desktop/mentoring
rm -f models/smoke.pt
git add scripts/train.py tests/test_train.py
git commit -m "feat: YOLO11s MPS 학습 스크립트 추가"
```

---

### Task 5: 영상 추론 (90프레임 간격 + 박스 유지)

**Files:**
- Create: `scripts/detect_video.py`
- Create: `tests/test_detect_video.py`

**Interfaces:**
- Consumes: `models/best.pt` (Task 4)
- Produces:
  - `scripts/detect_video.py` 안의:
    - `Detection` 데이터클래스 — 필드 `x1: int, y1: int, x2: int, y2: int, cls_id: int, conf: float`
    - `should_infer(frame_index: int, stride: int) -> bool`
    - `class_color(cls_id: int) -> tuple[int, int, int]` (BGR)
    - `parse_results(result, conf_threshold: float) -> list[Detection]`
    - `draw_detections(frame, detections: list[Detection], class_names: list[str]) -> Any`
    - `run(source, weights, stride, conf, save_path, show) -> dict`
  - 파일시스템(옵션): `outputs/<입력파일명>_detected.mp4`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_detect_video.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_detect_video.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'detect_video'`

- [ ] **Step 3: `scripts/detect_video.py` 구현**

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/test_detect_video.py -v
```
Expected: 14 passed.

- [ ] **Step 5: 전체 테스트 실행**

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load pytest tests/ -v
```
Expected: 48 passed (4 + 9 + 14 + 7 + 14).

- [ ] **Step 6: 실제 영상으로 확인**

사용자에게 테스트할 영상 파일 경로를 요청한다. 받은 뒤:

Run:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/detect_video.py --source <영상경로> --no-show
```
Expected: `frame 0: N개 탐지 ...` 형태 로그가 90프레임 간격으로 출력되고, 마지막에 `총 X프레임 처리, Y회 추론` 및 `결과 영상: outputs/<이름>_detected.mp4`. `Y == ceil(X / 90)` 인지 확인한다.

이어서 화면 표시까지 확인:
```bash
cd /Users/leehnsong/Desktop/mentoring && conda run -n yolo-load python scripts/detect_video.py --source <영상경로>
```
Expected: OpenCV 창이 열리고 적재물에 색상 박스와 `class 0.87` 형태 라벨이 표시된다. `q`로 종료된다.

- [ ] **Step 7: 커밋**

```bash
cd /Users/leehnsong/Desktop/mentoring
git add scripts/detect_video.py tests/test_detect_video.py
git commit -m "feat: 90프레임 간격 영상 추론 및 박스 오버레이 추가"
```

---

## 완료 기준

- [ ] `pytest tests/ -v` 전부 통과
- [ ] `datasets/merged/data.yaml`이 `nc: 3`, `names: [box, bicycle, stroller]`
- [ ] `models/best.pt`의 `model.names`가 세 클래스와 일치
- [ ] 실제 영상에서 90프레임 간격 추론이 확인되고 박스가 화면에 표시됨
- [ ] `git status`에 `.env`가 절대 나타나지 않음
