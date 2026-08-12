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
