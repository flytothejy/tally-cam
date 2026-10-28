# 벤치마크

모든 카운팅 방법을 **같은 골든셋, 같은 채점**으로 비교한다.

- `dataset.json` — 클립 목록과 조건(light · density · view), 분할(dev/test), 정답 파일, 기준선 파일
- `score.py` — 채점. 사건 단위(같은 방향 ±1초 1:1 매칭) 재현율·정밀도·F1, 총 대수 오차, 10초·60초 창별 대수 오차
- `preds/<방법>/<클립>.json|csv` — 각 방법의 출력(사건 목록). 업로드형 상용 제품처럼 구간 대수만 주는 경우는 창별 대수 오차만 비교
- `cameras.json` — 수집 캠페인 카메라 목록 (`scripts/record.py`가 사용)

## 규칙
1. test 분할의 정답은 방법 개발·튜닝 중에 보지 않는다. 학습·튜닝은 dev만.
2. test에는 dev에 없는 카메라를 반드시 포함한다(다른 도로로 옮겼을 때의 성능).
3. 결과는 조건별로 따로 보고한다. 평균 하나로 합치지 않는다.
4. 속도는 같은 장비에서 프레임당 ms로 함께 적는다.
5. 정답의 10%는 두 번 라벨링해 사람 간 일치도를 잰다 — 이보다 정밀한 차이는 의미 없음.

## 수집 캠페인 (2026-11-02 ~ 11-06)
평일 하루 5회(07:28 · 08:28 · 12:28 · 17:58 · 21:58) 카메라 8곳 × 120초.
PC에서 `scripts/record.py list → rec --part 0/3, 1/3, 2/3 → retry` 순서로 실행(예약 작업).
결과: `samples/campaign/<날짜>/<시각>/<카메라>.mp4`, `manifest.json` (깃에는 올리지 않음).

## 실행 예
    python3 bench/score.py --gt eval/gt/dongtan_night_02_v2.csv --pred run.json
    python3 bench/score.py --dataset bench/dataset.json --method yolo11s-bytetrack-v0 --split test
