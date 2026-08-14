# SDD Progress — YOLO 적재물 detection

Plan: `docs/superpowers/plans/2026-08-12-yolo-loading-detection.md`
Spec: `docs/superpowers/specs/2026-07-10-yolo-loading-detection-design.md`
Branch: `feat/yolo-loading-detection`
Base commit: 6dfb9ba

## Tasks
- Task 1: conda 환경 + 스캐폴드 + MPS 검증 — **COMPLETE**
- Task 2: 데이터셋 설정 + Roboflow 다운로드 — **COMPLETE**
- Task 3: 3클래스 데이터셋 병합 — **COMPLETE**
- Task 4: 학습 스크립트 구현 — **COMPLETE** (커밋 6b5f15a, 47/47 테스트)
- Task 4b: **본 학습 실행 — COMPLETE** (40 epoch / 9.18시간, 2026-08-14 05:33 완료)
- Task 5: 90프레임 간격 영상 추론 — **COMPLETE** (커밋 be5462c, 6143930)
- 최종 전체 브랜치 코드 리뷰 — **COMPLETE** (수정 반영, 테스트 67개)

## 재개 방법

브랜치는 기능적으로 완성 상태이며, 최종 리뷰 지적사항 반영까지 끝난 상태에서
재개한다 (Task 3부터 다시 시작할 필요 없음).

```bash
cd /Users/leehnsong/Desktop/mentoring
git checkout feat/yolo-loading-detection
conda activate yolo-load
pytest tests/ -v          # 67/67 통과해야 정상
```

## 확정된 사실 (재조사 불필요)

- conda 환경 `yolo-load` (Python 3.11.15) 생성 완료. **MPS 실동작 검증됨.**
- `.env`에 실제 Roboflow API 키 있음 (gitignored). `.env.example`은 플레이스홀더.
  키가 커밋된 적은 없음 — 히스토리 확인 완료.
- Roboflow 6개 소스 전부 다운로드 성공 (721MB). **슬러그 교체 불필요했음.**
- `configs/datasets.yaml`의 `class_map` 키가 실제 데이터셋 클래스명과 전부 일치함을 검증 완료
  (`Hand`, `0-human`/`1-wheelchair`/`2-suitcase`, 자전거 부품 `wheel`/`seat`/`chair` 등이 올바르게 제외됨).

## 다운로드된 데이터 현황

| source | 기여 클래스 | train 이미지 | valid 이미지 |
|---|---|---:|---:|
| box_cardboard_main | box | 5,877 | 280 |
| box_open_closed | box (Hand 제외) | 261 | 83 |
| mixed_stroller_bicycle | stroller, bicycle | 3,348 | **0** |
| stroller_ultimate | stroller | 1,030 | 167 |
| bicycle_bike_identification | bicycle (부품 제외) | 170 | 44 |
| bicycle_training_data | bicycle | 88 | 25 |

학습 인스턴스(박스) 수: **box 7,760 / bicycle 322 / stroller 4,772**

## 다음 세션에서 반드시 판단해야 할 것

1. **bicycle 클래스가 심각하게 적다 (322 인스턴스 vs box 7,760).**
   Task 3 병합 후 실제 클래스별 박스 수를 확인하고, 자전거가 여전히 적으면
   Roboflow에서 자전거 데이터셋을 추가로 받아 `configs/datasets.yaml`에 넣고
   `download_datasets.py` → `merge_datasets.py`를 다시 돌릴 것.
   COCO에 bicycle이 있으므로 대안으로 COCO 자전거 서브셋 사용도 검토 가능.
2. **`mixed_stroller_bicycle`은 valid split이 없다 (train only).**
   Task 3의 `merge_datasets.py`는 `_find_split_dir`가 `valid/`를 못 찾으면
   그 split을 건너뛰므로 동작은 하지만, 이 소스의 3,348장이 전부 train으로만 들어간다.
   val 세트가 stroller/bicycle을 충분히 대표하는지 병합 후 확인할 것.
3. **Task 4의 `test_resolve_device_returns_mps_when_available`**은 MPS 미지원 시
   아무것도 assert하지 않는 조건부 테스트다. Task 1에서 MPS 가용성이 이미 검증됐으므로
   무조건 `assert tr.resolve_device("mps") == "mps"` 로 구현하도록 구현자에게 지시할 것.

## 학습 시작하기 (사용자가 지시하면 실행)

**결정된 설정: YOLO11s, 40 epoch, batch 16, imgsz 640** (사용자 승인 완료 2026-08-13)

```bash
cd /Users/leehnsong/Desktop/mentoring
/opt/homebrew/anaconda3/envs/yolo-load/bin/python -u scripts/train.py \
  --epochs 40 --batch 16 --name loadobj \
  > /tmp/train_loadobj.log 2>&1
```

- **`conda run`을 쓰지 말 것** — 출력을 버퍼링하고 데드락이 난다. 환경 python을 직접 호출한다.
- 반드시 백그라운드로 실행하고 로그를 폴링한다:
  `tail -c 2000 /tmp/train_loadobj.log | LC_ALL=C tr '\r' '\n' | tail -15`
- **멈춤 감지:** 로그가 안 늘어나면 `ps -p <PID> -o etime,time,%cpu`. %CPU가 0이고 CPU time이
  안 오르면 학습이 아니라 hang이다.
- 완료 후 `models/best.pt` 생성 확인 → 클래스명이 `[box, bicycle, stroller]`인지 검증.
- **가장 중요한 산출물: 클래스별 mAP50.** 자전거 mAP가 낮으면 실제 데이터 추가가 필요하다.

### 실측 성능 (2026-08-13 측정)
- 1.4 it/s, epoch당 788 iteration (12,608장 / batch 16)
- **1 epoch ≈ 10분** (학습 9.4분 + 검증)
- 40 epoch ≈ **6.7시간** (patience=15로 더 일찍 끝날 수 있음)
- GPU_mem 8.43GB / 48GB — batch를 더 올릴 여유는 있음

### 알려진 함정 (겪은 문제)
1. `yolo11s.pt` 자동 다운로드가 HTTPS 읽기에서 무한 정지했다. 이미 curl로 받아
   레포 루트에 두었으므로(`/*.pt`로 gitignore됨) 네트워크를 타지 않는다. **삭제하지 말 것.**
2. subagent에 장시간 학습을 맡기면 subagent 종료 시 자식 프로세스가 같이 죽는다.
   메인 세션에서 `run_in_background`로 직접 실행할 것.

## Minor findings roll-up (최종 리뷰에서 판단)

- `scripts/download_datasets.py:94-95` — `get_api_key()`가 `load_dotenv` 후 `os.environ`을
  읽으므로 셸에서 export한 값도 허용된다. "오직 .env에서만"이라는 제약보다 느슨함.
  python-dotenv 표준 사용법이라 실무상 문제는 아님.
- `configs/datasets.yaml` / `download_datasets.py` — split 완전성(valid 존재 여부)을
  검증하지 않고 조용히 수용한다. Task 3에 영향.

## Log
Task 1: complete (commits 6dfb9ba..a383a52, review clean after 1 fix round; 4/4 tests, MPS verified)
Task 2: complete (commit 2d3cab9, review approved with no Critical/Important; 13/13 tests; COVERAGE_OK)
Task 3: complete (commits e1eb599..f89f4a7, review approved after 1 fix round; 30/30 tests, MERGE_OK; box=8176 bicycle=405 stroller=4948)
Task 3b (수정안): complete (commit 93047cd, review approved; 40/40 tests). 학습셋 오버샘플링 bicycle 8x -> train box=7760 bicycle=2576 stroller=4919, val은 원본 유지(dup=0, box 416 / bicycle 83 / stroller 176)
Task 4b (본 학습): complete. 40 epoch, Apple M-series MPS, 9.18시간 소요, 2026-08-14 05:33 완료.
  검증 결과 — mAP50: 전체 0.971 / box 0.994 / bicycle 0.922 / stroller 0.995. 추론 속도 6.2ms/이미지.
  가장 약한 지표는 bicycle Recall 0.892. `models/best.pt` 생성 후
  `models/best_20260814_loadobj_map50_0971.pt`로 백업 완료 (둘 다 gitignored, 삭제·덮어쓰기 금지).
  실환경 검증: 아파트 계단실 사진에서 bicycle·stroller는 탐지했으나 box는 conf=0.02까지
  낮춰도 0건 탐지 (검증셋 5/5 및 밝은 복도 사진(0.93, 0.84)에서는 box 탐지 성공). 코드 결함이
  아니라 학습 데이터와 실제 현장 이미지 간 도메인 갭으로 판단 — 개선하려면 실제 현장 이미지에
  라벨링이 필요하다.
Task 5 (90프레임 추론): complete (commits be5462c, 6143930).
최종 리뷰: 지적사항 7건 반영 (가중치 덮어쓰기 방지 --force, 빈 병합 시 exit 1, 데이터셋 버전 고정,
  progress print 가드, 취약 테스트 교체, README/PROGRESS 갱신). 테스트 67개로 증가.
Task 4 (코드): complete (commit 6b5f15a, 47/47 tests; 조건부 테스트를 무조건 assert로 수정). 본 학습은 사용자 요청으로 보류 — 지시 시 40 epoch 실행.
Task 4 리뷰: Approved (Critical/Important 0건). MPS fallback 환경변수가 torch import 이전에 설정됨을 정적 검증 완료. /*.pt gitignore 범위도 정확.
정리: 병합 경합으로 생긴 'datasets/merged 2/' (라벨 42개, 전부 중복 확인 후) 삭제, runs/smoke·models/smoke.pt 삭제. 데이터셋 무결성 재확인 (train 12605 / val 579).
