# SDD Progress — YOLO 적재물 detection

Plan: `docs/superpowers/plans/2026-08-12-yolo-loading-detection.md`
Spec: `docs/superpowers/specs/2026-07-10-yolo-loading-detection-design.md`
Branch: `feat/yolo-loading-detection`
Base commit: 6dfb9ba

## Tasks
- Task 1: conda 환경 + 스캐폴드 + MPS 검증 — **COMPLETE**
- Task 2: 데이터셋 설정 + Roboflow 다운로드 — **COMPLETE**
- Task 3: 3클래스 데이터셋 병합 — pending (다음에 여기서 재개)
- Task 4: YOLO11s 학습 (MPS) — pending
- Task 5: 90프레임 간격 영상 추론 — pending
- 최종 전체 브랜치 코드 리뷰 — pending

## 재개 방법

```bash
cd /Users/leehnsong/Desktop/mentoring
git checkout feat/yolo-loading-detection
conda activate yolo-load
pytest tests/ -v          # 13/13 통과해야 정상
```

그 다음 Task 3부터 진행. 브리프 생성:
```bash
/Users/leehnsong/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-12-yolo-loading-detection.md 3
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

## Minor findings roll-up (최종 리뷰에서 판단)

- `scripts/download_datasets.py:94-95` — `get_api_key()`가 `load_dotenv` 후 `os.environ`을
  읽으므로 셸에서 export한 값도 허용된다. "오직 .env에서만"이라는 제약보다 느슨함.
  python-dotenv 표준 사용법이라 실무상 문제는 아님.
- `configs/datasets.yaml` / `download_datasets.py` — split 완전성(valid 존재 여부)을
  검증하지 않고 조용히 수용한다. Task 3에 영향.

## Log
Task 1: complete (commits 6dfb9ba..a383a52, review clean after 1 fix round; 4/4 tests, MPS verified)
Task 2: complete (commit 2d3cab9, review approved with no Critical/Important; 13/13 tests; COVERAGE_OK)
