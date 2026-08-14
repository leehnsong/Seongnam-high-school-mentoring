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

### 사전 준비: `yolo11s.pt`

`yolo11s.pt`(YOLO11s 사전학습 가중치)는 gitignore 대상이라 새로 클론한 저장소에는
없습니다. Ultralytics가 첫 학습 시 자동으로 다운로드하지만, 이 환경에서는 그 다운로드가
무한정 멈추는 현상이 관찰되었습니다. 학습이 첫 epoch 시작 전에 멈춘 것처럼 보이고
CPU 사용률이 0%라면, 직접 받아 저장소 루트에 두세요:

```bash
curl -L --retry 3 --connect-timeout 15 -o yolo11s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

`scripts/train.py`는 이 파일명을 현재 작업 디렉토리 기준 상대 경로로 찾으므로,
반드시 저장소 루트에서 실행하세요.

### 재학습 시 기존 모델 보호

`scripts/train.py`는 기본적으로 `models/best.pt`가 이미 있으면 덮어쓰지 않고
`FileExistsError`로 중단합니다. 의도적으로 재학습 결과를 덮어쓰려면 `--force`를 붙이거나
`--dest`로 다른 경로를 지정하세요.

```bash
python scripts/train.py --force
```

### 영상 추론: `--no-show`

화면 표시 없이(헤드리스 환경, 서버 등) 실행하려면 `--no-show`를 붙이세요.

```bash
python scripts/detect_video.py --source 영상.mp4 --no-show
```

### 90프레임 간격 추론

`detect_video.py`는 매 프레임을 추론하지 않고 기본 90프레임(약 3초, 30fps 기준)마다
한 번만 YOLO 추론을 실행합니다. 적재물(박스/자전거/유모차)은 대부분 정지해 있으므로,
추론하지 않은 프레임에는 직전 추론 결과의 박스를 그대로 다시 그려 보여줍니다.
`--stride`로 간격을 조절할 수 있습니다.

## 테스트

```bash
pytest tests/ -v
```

## 학습 결과 (2026-08-14 측정)

- 40 epoch, Apple M-series MPS 기준 9.18시간 학습
- 검증 mAP50: 전체 0.971 / box 0.994 / bicycle 0.922 / stroller 0.995
- 추론 속도: 이미지당 6.2ms

## 알려진 한계

여러 개의 단일 클래스 데이터셋을 병합하기 때문에, 예를 들어 박스 데이터셋 이미지에
자전거가 찍혀 있어도 라벨이 없습니다. 이런 **미라벨 인스턴스**는 학습 시 약한 오답 신호가
됩니다. 클래스가 함께 등장하는 이미지가 많다고 판단되면 해당 이미지를 제외하거나
직접 라벨을 보강하세요.
