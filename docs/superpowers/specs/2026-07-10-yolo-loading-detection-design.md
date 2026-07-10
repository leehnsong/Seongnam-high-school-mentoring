# YOLO 적재물 Detection 프로그램 — 설계 문서

작성일: 2026-07-10

## 1. 목적

적재물이 있으면 안 되는 공간(예: 복도·공용공간)을 촬영한 **녹화 영상 파일**을 분석하여,
프레임에 나타난 적재물의 **종류(class)와 위치(bounding box)** 를 화면에 표시한다.

감지 대상 클래스 (3종):

| id | class | 정의 |
|----|-------|------|
| 0 | `box` | 박스처럼 생긴 것 전부 (골판지/플라스틱/나무 등 무관) |
| 1 | `bicycle` | 일반 자전거만 (오토바이·킥보드 제외) |
| 2 | `stroller` | 아기 유모차만 (쇼핑카트·손수레 제외) |

## 2. 실행 환경

- **하드웨어**: Apple Silicon Mac (arm64), 통합 메모리 48GB
- **가속기**: PyTorch **MPS** (CUDA 아님)
- **Python 관리**: **conda** 가상환경 (Python 3.11 권장)
- **학습·추론 모두 로컬**에서 실행
- Roboflow API 키는 `.env` 파일에 저장하고 스크립트가 읽음 (git 제외, 채팅/코드에 하드코딩 금지)

## 3. 모델 전략

- **방식 A: 3개 클래스를 하나의 커스텀 모델로 통합 학습** (채택)
  - box, bicycle, stroller를 한 데이터셋으로 합쳐 YOLO 하나를 fine-tuning
  - 자전거는 COCO에 이미 있으나, 관리·코드 단순화를 위해 통합 모델로 재학습
- **베이스 모델**: **YOLO11s** (Ultralytics)
  - 녹화 영상(비실시간) 분석이라 정확도를 챙기면서 MPS에서 무리 없이 동작
  - 더 가볍게 가려면 YOLO11n으로 교체 가능

## 4. 아키텍처 / 프로젝트 구조

파이프라인은 독립적인 4단계 스크립트로 분리 — 각각 따로 테스트·재실행 가능.

```
① download → ② merge → ③ train → ④ detect_video
```

```
mentoring/
├── environment.yml          # conda 환경 정의
├── .env                     # ROBOFLOW_API_KEY (gitignore)
├── .gitignore
├── requirements.txt         # pip 의존성 (ultralytics 등)
├── README.md
├── configs/
│   └── data.yaml            # 3클래스 통합 데이터셋 정의 (merge가 생성)
├── scripts/
│   ├── download_datasets.py # Roboflow에서 3개 데이터셋 다운로드
│   ├── merge_datasets.py    # 클래스ID 재매핑 → 하나로 병합
│   ├── train.py             # YOLO11s fine-tuning (device=mps)
│   └── detect_video.py      # 90프레임마다 추론 + 박스 그리기
├── datasets/
│   ├── raw/{box,bicycle,stroller}/   # 다운로드 원본
│   └── merged/                       # 최종 학습셋
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── data.yaml
├── models/
│   ├── yolo11s.pt           # 사전학습 베이스 (자동 다운로드)
│   └── best.pt              # 학습 완료 모델
└── outputs/                 # 결과 영상/프레임
```

각 단위의 책임:

- **download_datasets.py** — 입력: Roboflow API 키 / 출력: `datasets/raw/{box,bicycle,stroller}` (각 YOLO 포맷). 의존: `roboflow`, `.env`.
- **merge_datasets.py** — 입력: `datasets/raw/*` / 출력: `datasets/merged/` + `data.yaml`. 의존: 없음(표준 라이브러리 중심).
- **train.py** — 입력: `datasets/merged/data.yaml`, `yolo11s.pt` / 출력: `models/best.pt`. 의존: `ultralytics`.
- **detect_video.py** — 입력: 영상 파일, `models/best.pt` / 출력: 화면 표시 + `outputs/` 저장. 의존: `ultralytics`, `opencv`.

## 5. 데이터 흐름

1. **download**: Roboflow Universe에서 box / bicycle / stroller 각각의 공개 데이터셋을 YOLO 포맷으로 `datasets/raw/`에 받는다.
   - 자전거는 오토바이·킥보드가 섞이지 않은 셋을 선별, 유모차는 아기 유모차 셋을 선별한다.
2. **merge**: 각 데이터셋의 클래스 번호를 통합 규칙(`box=0, bicycle=1, stroller=2`)으로 **재매핑**하고,
   라벨 없는 이미지·깨진 라벨은 건너뛰며 `datasets/merged/`에 `train`/`val`로 합친다. 최종 `data.yaml`을 생성한다.
3. **train**: `yolo11s.pt`에서 fine-tuning.
   - `device=mps`, `imgsz=640`, **`batch=16`** (48GB 메모리, 필요 시 32까지), `epochs≈80` + early stop(`patience`).
   - 학습 결과 `runs/.../best.pt`를 `models/best.pt`로 복사.
4. **detect_video**: `models/best.pt` 로드 → 영상 프레임 순회 → **90프레임마다만 추론**.
   - 적재물은 정지 물체이므로 **추론 사이의 89프레임에는 마지막 감지 결과(박스)를 그대로 유지**해 화면 표시가 끊기지 않게 한다.
   - 클래스별 색상 박스 + 라벨(class명 + confidence) 표시. `q`로 종료, 결과 영상 `outputs/`에 저장(옵션).

## 6. 에러 처리

- **키 없음/다운로드 실패**: `.env`에 `ROBOFLOW_API_KEY` 없으면 명확한 안내 후 중단. 네트워크 실패 시 재시도 안내.
- **MPS 미지원 연산**: `PYTORCH_ENABLE_MPS_FALLBACK=1` 설정. MPS 사용 불가 시 CPU로 폴백.
- **병합 단계**: 라벨 없는/형식 오류 이미지는 건너뛰고 카운트를 로그로 남긴다.
- **영상 열기 실패**(코덱 등): 명확한 에러 메시지 후 종료.

## 7. 테스트 (스모크 테스트)

- 병합 후 `data.yaml`의 경로·클래스 수(=3)·이미지 개수 검증.
- **1 epoch 짧은 학습**으로 파이프라인 정상 동작 확인 후 본 학습 진행.
- 샘플 클립 하나로 `detect_video`의 90프레임 추론·박스 유지 동작 확인.

## 8. 범위 밖 (YAGNI)

- 실시간 스트림/CCTV 처리 (녹화 파일만 대상)
- 알림·통계 집계·DB 저장
- 웹/GUI 프론트엔드 (OpenCV 창 표시로 충분)
- 객체 추적(tracking) — 정지 물체라 단순 박스 유지로 충분
