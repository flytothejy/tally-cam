# tally-cam "퀀텀 점프" 후보 조사 보고서 — 범용 선 통과 계수·추적 (첫 도메인: 고속도로 CCTV 차량)

가장 큰 개선 후보는 지금의 "전역 검출 → 추적 → 선분 통과" 사슬을 더 다듬는 것이 아니라, **이미 가진 점 라벨 1,651개를 그대로 학습 신호로 쓰는 "시공간 띠 위 학습형 점(히트맵) 검출기"를 주 계수기로 세우고, 검출+추적 경로는 보조 증거로 융합하며, 클라우드의 오픈 어휘 모델(SAM 3·CountGD++·CountVid)은 실시간 판정이 아니라 오토 라벨러·불일치 확인기로 쓰는 구조**입니다. 남은 오류의 대부분이 "검출기가 못 잡은 차"(재현율 78~91%, 정밀도 94~99%)입니다. 추적기를 바꾸는 시도는 이미 85~87로 실패했으므로 기대 효과가 작고, 병목은 검출 신호와 기준선 주변의 국소 증거 쪽에 있습니다.

## TL;DR
- **1순위 실험: 시공간 띠에서 통과 사건을 점·히트맵으로 직접 검출하는 소형 CNN을 사람 점 라벨로 학습.** 시공간 띠(visual rhythm) 위에 YOLOv8을 돌린 Ribeiro & Hirata(SIBGRAPI 2023, arXiv 2501.04534)는 띠 이미지 33장 라벨만으로 "평균 계수 정확도 약 99.15%, 추적 기반 방식보다 3배 빠른 처리 속도"(초록 원문)를 보고했습니다(ByteTrack 기준선 56 FPS 대비 186 FPS). DRIVE-TSI 연구(IEEE, 2021)는 주간 도심에서 97.95%·2.91 ms/프레임을 보고했습니다.\[1\] 다만 둘 다 위에서 내려다보는 단방향·주간 조건이어서, 우리 조건(야간·서다 가다·가림)에서의 성능은 미확인이며 직접 재야 합니다.
- **2순위: 클라우드 파운데이션 모델로 새 200개 클립에 의사 박스를 만들어 엣지 검출기를 미세조정.** COCO 사전학습 모델을 도메인에 미세조정만 해도 계수 오차가 크게 준 사례가 반복해서 보고됩니다(공개 저장소 기준 YOLO26s COCO 9.7% → 미세조정 5.7%). 상용화를 생각하면 엣지 검출기는 AGPL인 Ultralytics YOLO11/YOLO26 대신 Apache-2.0인 RF-DETR(N/S/M/L)이나 D-FINE으로 옮기는 것이 유리합니다.
- **범용 구조 관점: "대상 지정(텍스트·예시·점) → 클라우드 범용 모델(오토 라벨/확인) → 엣지 소형 모델(증류) → 선·영역·띠 기반 계수 규칙 → 사건 단위 평가"는 군중 LOI, 소나 물고기 계수(Caltech Fish Counting), 영상 개체 계수(VIC)에서 이미 같은 구조로 쓰이고 있어 도메인 이식성이 높습니다.** 반면 하나의 모델로 차량·사람·동물을 미세조정 없이 사건 단위로 정확히 세는 시스템은 아직 없습니다. 공개된 범용 영상 계수기(CountVid)도 "영상 전체 고유 개체 수"만 셀 뿐, 선 통과·방향별 집계는 하지 않습니다.

## 1. 한 페이지 요약: 가장 큰 개선을 줄 후보 5개

| 순위 | 후보 | 공략 오류 | 왜 크게 기대하는가 | 범용성 |
|---|---|---|---|---|
| 1 | **시공간 띠 학습형 점/히트맵 검출기** (CenterNet식 히트맵 또는 P2PNet식 점 회귀, 띠 이미지 입력) | 1, 3, 4, 5 | 라벨 형식(띠 위의 점)과 출력이 정확히 일치. 규칙 기반 덩어리 세기(82~85)의 한계(쪼개짐·합쳐짐)를 학습으로 대체. 띠 위 학습형 검출은 차량에서 97.95~99.15% 정확도 보고. 추적 불필요 → CPU 비용 매우 낮음 | 높음(물고기 에코그램, 군중 LOI와 같은 구조) |
| 2 | **사건 단위 늦은 융합 + 불일치 캐스케이드** (띠 검출기 사건 ∪ 검출+추적 통과 사건, 헝가리안 ±1초 매칭으로 합치고 한쪽만 본 사건은 클라우드 확인) | 1, 3, 6 | 두 경로의 놓침이 서로 다른 원인(띠: 정지·쪼개짐 / 추적: 가림·박스 합침)이어서 합집합이 재현율을 올리고, 불일치만 클라우드로 보내면 비용이 작음 | 높음 |
| 3 | **오토 라벨러로 의사 박스 생성 → 엣지 검출기 미세조정** (SAM 3·CountGD++·CountVid → RF-DETR-N/S 또는 D-FINE-N/S) | 1, 2, 6 | 현재 검출기는 COCO 사전학습 그대로. 720×480·1 Mbps·야간 헤드라이트는 COCO 분포 밖. 미세조정이 계수 오차를 크게 줄인 공개 사례 존재 | 높음(새 도메인도 같은 절차) |
| 4 | **기준선 주변 ROI 확대·슬라이싱 + 야간 광원 전용 경로** (기준선 띠 주변만 2~3배 업샘플해 검출, 야간엔 헤드라이트 덩어리를 띠 위 별도 클래스로 검출) | 2 | 먼 쪽 작은 헤드라이트 차 재현율 68%가 가장 큰 단일 오류. SAHI(Akyon 외, ICIP 2022)는 VisDrone·xView에서 검출기 재학습 없이 AP를 FCOS/VFNet/TOOD 기준 +6.8/+5.1/+5.3%, 슬라이스 미세조정까지 하면 누적 +12.7/+13.4/+14.5% 올렸음. 기준선 주변만 확대하면 비용이 전체 슬라이싱보다 훨씬 작음 | 중간(ROI 확대는 범용, 헤드라이트 규칙은 차량 전용) |
| 5 | **교통공학식 가상 루프(점유 on/off) 판정을 띠 검출기의 후처리로** (차로별 점유 시계열 + 최소 차두 시간 제약으로 쪼개짐 병합·합침 분리) | 4, 5 | 서다 가다에서 한 차가 여러 덩어리로 쪼개지는 문제는 "같은 차로에서 물리적으로 가능한 최소 차두 간격" 제약으로 줄일 수 있음. 학습 없이 규칙 교체만으로 벤치에서 빠르게 확인 가능 | 중간(차로·흐름 개념은 사람·물고기에도 적용되나 파라미터는 도메인별) |

**하지 말 것(근거 있는 비추천)**: 추적기만 교체(OC-SORT·Deep OC-SORT·Hybrid-SORT 등)하는 실험에 많은 시간을 쓰는 것. 같은 검출기를 쓰면 추적기 간 계수 오차 차이는 1%p 안팎으로 보고됩니다(공개 저장소의 UA-DETRAC 평가에서 FastTracker 3.1%, OC-SORT 3.3%, ByteTrack은 최고와 1%p 이내). 게다가 우리 오류는 검출 단계의 놓침이 대부분입니다. 다만 검출 캐시가 있으니 비용이 적은 확인 실험으로는 넣어 볼 만합니다.

## 2. 후보 표 (16개)

표가 넓어 A(정체·비용)와 B(성능·시험·효과·위험)로 나눴습니다. 번호로 대응합니다. "미확인"은 원문 표·README에서 직접 확인하지 못한 숫자입니다.

### 표 A — 방법·범용성·라벨·코드·비용

| # | 방법 | 한 줄 설명 | 공략 오류 | 범용성 | 필요 라벨 | 공개 코드 / 라이선스 / 가중치 | 연산 비용 |
|---|---|---|---|---|---|---|---|
| 1 | 띠 위 학습형 히트맵/점 검출 (자체 구현, CenterNet·P2PNet 방식) | 시공간 띠 이미지에서 통과 사건을 (시각, 위치) 점으로 직접 검출 | 1,3,4,5 | 높음 — 어떤 대상이든 선을 지나면 띠에 흔적. 라벨도 점 | 점(보유 1,651개) | P2PNet: 공식 GitHub(TencentYoutuResearch, ICCV 2021), 라이선스 파일 존재·종류 미확인, VGG16 기반 | 띠 이미지는 (선 길이 × 시간)으로 작아 CPU에서 수 ms~수십 ms 예상(미확인, 직접 측정) |
| 2 | Visual Rhythm + YOLO (Ribeiro & Hirata, SIBGRAPI 2023) | 900프레임(30초) 단위 띠에서 YOLOv8s로 "자국" 검출 → 통과 프레임만 분류 | 1,4 | 높음 | 띠 박스 33장(증강 후 79장) | github.com/victor-nasc/Vehicle-Counting, 라이선스 미확인; YOLOv8은 AGPL | 186 FPS(논문 보고)\[2\] |
| 3 | DRIVE-TSI 시공간 영상 검출 (IEEE, 2021) | TSI 위 검출 + 거의 정지한 차의 왜곡을 잇는 Cross-Image Object Linking | 4 | 중간~높음 | TSI 박스(자동 생성 데이터셋) | 미확인 | 2.91 ms/프레임(논문 보고)\[1\] |
| 4 | TM-Net (IEEE Access 2021, Yang 외) | TSI 위 밀도맵 회귀(점 라벨 → 밀도) | 1,3 | 높음 | 점 | 미확인 | 미확인\[3\] |
| 5 | 에코그램 계수 회귀 (Van Brunt 외, arXiv 2502.05129) | 200프레임 소나 영상을 2D 에코그램으로 압축해 ResNet-18이 상·하류 수를 직접 회귀 | 참고(구조 이식 근거) | 높음 — 차량 띠와 동일 구조 | 개수만 + 약지도(검출기·추적기 자동 라벨) | 미확인 | ResNet-18, 매우 낮음\[4\]\[5\] |
| 6 | 사건 단위 늦은 융합 + 불일치 캐스케이드 (자체 설계) | 띠 경로·추적 경로 사건을 ±1초 매칭으로 합치고, 한쪽만 본 사건만 클라우드 확인 | 1,3,6 | 높음 | 없음(평가에만 점) | 자체 | 추가 비용 ≈ 0 + 불일치 사건당 클라우드 호출 |
| 7 | SAHI / 기준선 ROI 확대 | 이미지를 겹치는 조각으로 잘라 확대 추론 후 병합. 우리는 기준선 주변만 | 2 | 높음(검출기 무관) | 없음(추론만), 미세조정 시 박스 | github.com/obss/sahi (ICIP 2022), 라이선스 미확인 | 조각 수 × 검출기 비용. ASAHI는 SAHI 대비 추론시간 20~25% 감소 보고\[6\]\[7\] |
| 8 | 야간 헤드라이트/후미등 쌍 검출 (Lashkov 외 2024 CACIE, ESWA 2024 GMM 쌍 짓기 등) | 밝은 덩어리 분할 → 쌍 짓기 → 추적·계수 | 2 | 낮음(차량 전용) | 없음~소량 | 공개 코드 미확인 | 매우 낮음(고전 영상처리) |
| 9 | 주간→야간 스타일 변환 증강 (arXiv 2412.16478) | 주간 라벨을 야간 스타일로 변환해 야간 검출 학습 | 2 | 중간 | 기존 주간 라벨 재사용 | 미확인 | 학습 시에만 |
| 10 | RF-DETR (ICLR 2026) | DINOv2 백본 실시간 DETR, NMS 없음, 미세조정 지향 | 1,2,6 | 높음(소량 미세조정으로 새 도메인) | 박스 | github.com/roboflow/rf-detr, N/S/M/L은 Apache-2.0, XL/2XL은 PML 1.0 | N: 30.5M 파라미터, 384px, T4 TensorRT FP16 2.3 ms; CPU·Jetson 지연 미확인\[8\]\[9\] |
| 11 | D-FINE / DEIM | 실시간 DETR 계열 | 1,2 | 높음 | 박스 | D-FINE Apache-2.0 | D-FINE-N 3.8M 파라미터, 2.1 ms(T4) — 단, 단순 FP16 양자화 시 AP 0.5로 붕괴 보고\[8\]\[10\] |
| 12 | YOLO26 (Ultralytics, 2025~2026) | NMS 없는 end-to-end, ProgLoss+STAL로 작은 물체 개선 주장 | 1,2 | 높음 | 박스 | AGPL-3.0 또는 Enterprise(유료) | YOLO26n: CPU ONNX 38.9 ms, COCO mAP 40.9(Ultralytics 문서)\[11\]\[12\] |
| 13 | 최신 추적기 (OC-SORT CVPR 2023, Deep OC-SORT ICIP 2023, Hybrid-SORT AAAI 2024) | 가림 후 복구·약한 단서로 연결 개선 | 3 | 높음 | 없음 | 공개 코드 있음(각 저장소), 라이선스 개별 확인 필요 | 낮음(Deep OC-SORT는 ReID 추가 비용) |
| 14 | SAM 3 (Meta, arXiv 2511.16719) | 텍스트·예시로 지정한 개념의 모든 인스턴스를 검출·분할·추적 | 6(오토 라벨), 1(확인기) | 매우 높음 | 없음(텍스트 "car" 또는 예시 박스) | 공개(github.com/facebookresearch/sam3, 가중치 포함), 라이선스 종류 미확인 | 848M 파라미터, GPU 필요. 동시 약 5개 객체에서 준실시간(2차 출처) |
| 15 | CountGD++ / CountVid (Oxford VGG, CVPR 2026 / AAAI 2026) | 텍스트+긍정·부정 예시로 세는 범용 계수기 / 이를 SAM 2.1과 결합한 영상 고유 개체 계수 | 6(오토 라벨), 1(확인기) | 매우 높음 | 없음~예시 몇 개 | CountVid·CountGD++ GitHub 공개, 라이선스 미확인 | Grounding DINO(Swin-T) + SAM 2.1, GPU 필요\[13\] |
| 16 | 영상 개체 계수 VIC (DRNet CVPR 2022, 약지도 VIC CVPR 2024, OMAN 2025) | 프레임 쌍 사이 "새로 들어온 개체(inflow)"를 최적 수송·매칭으로 추정해 누적 | 1,3 | 높음(약지도 VIC 데이터셋은 보행자·car·bus·van 포함) | 점 + inflow/outflow 표시 | 공개 코드 있음(OMAN 등), 라이선스 미확인 | 중간(GPU 권장)\[14\]\[15\] |

### 표 B — 보고 성능·우리 벤치 시험법·예상 효과·위험

| # | 보고된 성능(데이터셋·지표) | 우리 벤치에서 시험하는 구체적 방법 | 예상 효과와 근거 | 위험 |
|---|---|---|---|---|
| 1 | 직접 대응 논문 없음(2·3·4·5가 근거) | 띠 이미지(차로별)를 입력으로 1채널 히트맵+방향 2채널 출력의 소형 U-Net/CenterNet 학습. 라벨 점에 가우시안. 카메라 4곳 leave-one-camera-out 교차검증. 출력 피크 → 사건 → 기존 헝가리안 ±1초 채점 | **높음** — 라벨과 출력이 일치하고, 규칙 기반 띠(82~85)의 오류 원인(쪼개짐·배경 흔들림)을 학습이 흡수 | 점 1,651개·카메라 4곳으로 새 화각 일반화 부족. 정지 차량은 띠에서 가로로 긴 줄이 되어 "언제 지났나" 정의가 모호 |
| 2 | 상단 시점 공개 영상 4개: 초록 원문 "평균 계수 정확도 약 99.15%, 추적 기반보다 3배 빠름"(영상별 100/98.90/98.56%), ByteTrack 기준선 100/99.04/98.48%, 186 FPS vs 56 FPS | 우리 띠에 그대로 YOLO 박스 대신 점 학습으로 재현(1번의 기준선) | 중간 — 근거는 강하나 상단 시점·단방향·주간 | 측면/원근 시점·야간·정체 분석 없음 |
| 3 | "주간 도심 97.95% 계수 정확도, 2.91 ms/프레임"\[1\] | Cross-Image Linking 아이디어를 서다 가다 병합 규칙으로 차용 | 중간 — 오류 4 직접 표적 | 원문 전문 미확인. 야간 미평가로 보임 |
| 4 | 초록에 숫자 없음(UA-DETRAC 평가) | 밀도 회귀 변형을 1번과 비교(점 피크 vs 밀도 적분) | 중간 | 밀도 적분은 대수는 맞아도 사건 단위 F1 산출이 어려움 |
| 5 | Caltech Fish Counting: KL-val nMAE 23.0%(강+약 라벨), 분포 밖 KR 30.7%. 같은 데이터에서 검출+추적(YOLOv5m) 기준선은 4.9%/11.8%\[16\] | 교훈만 가져옴: "개수 직접 회귀"보다 "사건 검출"이 훨씬 정확 | 설계 근거로 높음 | 개수 회귀를 주 계수기로 쓰면 안 된다는 반례\[5\] |
| 6 | 직접 대응 논문 없음 | 기존 최고(94.1) 사건과 1번 사건을 ±1초 매칭. 합집합·교집합·신뢰도 가중 투표 비교. 불일치 사건 비율 = 클라우드 호출률로 기록 | **높음** — 재현율이 부족한 두 경로의 놓침 원인이 달라 합집합이 유리 | 합집합은 정밀도를 떨어뜨림 → 확인기 품질이 관건 |
| 7 | VisDrone/xView(Akyon 외, ICIP 2022): 추론만으로 AP +6.8/+5.1/+5.3%(FCOS/VFNet/TOOD), 슬라이스 미세조정 포함 누적 +12.7/+13.4/+14.5% | 기준선 주변 띠(예: 위아래 64px)만 잘라 2배 업샘플 후 기존 검출기로 추론. 야간 클립만 따로 채점 | **중간~높음**(오류 2) — 먼 쪽 작은 차가 바로 이 조건 | 1 Mbps 압축 블록 잡음이 확대되어 오검출 증가 가능 |
| 8 | 2017 IEEE 헤드라이트 궤적 매칭: 야간 검출률 81.19%.\[17\] 2024 연구들은 정량값 원문 미확인 | 야간 띠에서 밝은 점 쌍(같은 시각, 차폭 간격) 규칙 → 1번 검출기의 야간 보조 채널로 | 중간 — 오류 2에 직접 | 반사광·가로등·자막 오버레이 오검출. 차량 전용 |
| 9 | 원문 수치 미확인 | 주간 의사 라벨을 야간 스타일로 변환해 3번(미세조정)에 섞기 | 낮음~중간 | 합성-실제 격차 |
| 10 | COCO AP 48.0(N, 논문 초록; README는 48.4, 2.3 ms)·53.0(S, 3.5 ms)·56.5(L, 6.8 ms), T4 TensorRT FP16; 논문은 "nano가 비슷한 지연에서 D-FINE nano보다 5.3 AP 높다"고 보고; RF100-VL로 도메인 전이 평가 | 의사 박스로 RF-DETR-N/S 미세조정 → 검출 캐시 재생성 → 기존 세기 규칙 그대로 비교. CPU·Jetson 지연 직접 측정 | **높음**(라이선스+정확도) | 384~512px 입력이 원거리 소형 차에 불리할 수 있음. CPU 지연 미공개 |
| 11 | D-FINE-N COCO 42.7 AP, 2.1 ms\[8\] | 10번의 대안. 3.8M 파라미터로 Jetson 이하 후보 | 중간 | FP16 양자화 취약성 보고 |
| 12 | YOLO26n CPU ONNX 38.9 ms(YOLOv8n 80.4 ms), "CPU 최대 43% 빠름"(제조사 주장)\[12\] | 현재 YOLO11s 960을 YOLO26s로 바꿔 캐시 재생성(연구 단계 비교용) | 중간 | AGPL — 상용 배포 시 소스 공개 의무 또는 유료 라이선스 |
| 13 | DanceTrack HOTA: OC-SORT 54.6 → Hybrid-SORT 62.2; MOT17 Deep OC-SORT 64.9 HOTA. 계수 관점: 공개 저장소(UA-DETRAC) 추적기 간 계수 오차 차이 1%p 이내\[18\]\[19\]\[20\] | 검출 캐시 위에서 추적기만 교체, 촘촘 낮 클립만 따로 채점 | **낮음** — 이미 추적기 교체가 85~87로 실패, 놓침은 검출 단계 | 시간 소모 대비 이득 작음 |
| 14 | SA-Co/Gold cgF1 54.1(OWLv2 24.6), 인간 대비 약 74%. BDD100K(주행) val TETA: SAM 3 47.2 < "SAM 3 검출기 + 추적-by-검출" 51.0\[21\]\[22\] | 새 200개 클립 일부에 "car, truck, bus" 텍스트로 마스크·박스 생성 → 사람 점 라벨과 대조해 품질 측정 → 의사 라벨로 사용 | **높음**(오토 라벨러), 중간(실시간 확인기) | 1 Mbps 야간에서의 품질 미확인. 도로 장면에선 자체 추적보다 검출기+단순 추적이 나았다는 표 결과 |
| 15 | CountGD: FSC-147 MAE 5.74(텍스트+예시, 2차 출처).\[23\] CountVid: VideoCount(TAO·MOT20·펭귄·결정)에서 기존 대비 정확도 향상 보고 | 불일치 사건 구간(±2초) 크롭에 텍스트 "car" + 이 카메라의 예시 박스 1~3개로 개수 확인. 부정 예시(버스 그림자·자막)로 오검출 억제 | 중간 — 확인기로 유용, 실시간 아님 | GPU 비용·지연 미측정. 영상 전체 고유 수를 세며 선 통과는 미지원\[24\] |
| 16 | WuhanMetroCrowd 등 보행자 VIC; 약지도 VIC 데이터셋은 약 40만 개 inflow/outflow·박스 라벨(보행자·car·bus·van)\[15\]\[25\] | 사람 도메인 이식 시험에서 기준선 비교 대상 | 차량엔 낮음, 범용 구조 참고로 중간 | 추적 없이 "새로 나타난 개체"를 세는 구조라 방향별 선 통과 정의와 다름 |

## 3. 질문별 상세 (Q0~Q8)

### Q0. 범용 카운팅·추적의 지형
- **이미지 계수**: FamNet → CounTR → LOCA(밀도) → DAVE·GeCo(검출 기반) 순으로 발전했습니다. GeCo(Pelhan·Lukežič·Zavrtanik·Kristan, 류블랴나 대학, NeurIPS 2024)는 "선두 few-shot 검출 기반 계수기보다 총 계수 MAE가 ∼25% 낮다"고 보고했고, 검출 기반 계수기가 밀도 기반을 처음으로 넘었다고 밝혔습니다. GeCo2(AAAI 2026)는 MCAC에서 LOCA보다 MAE를 27% 줄였다고 보고했습니다. 함의: **범용 계수도 "점·박스로 위치를 내는" 쪽으로 수렴** — 우리 사건 단위 F1과 같은 방향입니다.
- **오픈 어휘**: CountGD(NeurIPS 2024)는 텍스트·예시·둘 다로 대상을 지정합니다. CountGD++(arXiv 2512.23351, CVPR 2026)는 세 가지를 더했습니다. "세지 말 것"을 부정 프롬프트로 지정하고, 텍스트만으로 1차 검출한 상위 박스를 '의사 예시'로 되먹이며, 외부 이미지(합성 포함)의 예시도 받습니다. "예시 몇 개로 대상 지정"이라는 tally-cam 원칙에 정확히 맞는 인터페이스입니다.
- **영상**: CountVid(AAAI 2026)는 CountGD-Box 검출 + SAM 2.1 추적으로 영상 전체의 고유 개체 수를 셉니다.\[26\] SAM 3(2025-11)는 개념 프롬프트 하나로 모든 인스턴스를 검출·분할·추적하는 848M 모델입니다.\[27\]\[28\] **하지만 어느 것도 "선 통과·방향별"을 기본으로 하지 않습니다** — 계수 규칙 층은 우리가 가져야 합니다.
- **분야 간 공통 구조(통과량)**: 네 분야가 같은 구조를 씁니다.
  - 군중 LOI: Ma & Chan(CVPR 2013)은 시간 단면 + 정수계획, Zhao 외(ECCV 2016)는 밀도 × 속도 맵.
  - 소나 물고기: Caltech Fish Counting(ECCV 2022)은 가상 중앙선 통과를 세는 현장 절차를 그대로 평가 프로토콜로 채택했고, 남은 오류의 대부분이 트랙 끊김에 의한 과소 계수였습니다.
  - 보행자 VIC: inflow 누적.
  - 교통: AI City Track 1의 tripwire·진입/진출 영역.
  
  **트랙 끊김 → 과소 계수라는 실패 양상이 물고기와 우리 차량에서 똑같이 나타난다**는 점이 구조적 일반화의 근거입니다.
- **미세조정 없는 성능 하락·최소 라벨**: 같은 물고기 데이터에서 새 장소(분포 밖 강)로 가면 오차가 1.3~2.4배 커졌습니다. 개수 회귀는 23.0% → 30.7%(nMAE), 검출+추적 기준선은 4.9% → 11.8%였습니다. 오픈 어휘 계수기에서 예시 1~3개로 대상을 지정하는 것은 표준입니다. 하지만 사건 단위 선 통과에 필요한 최소 점 라벨 수를 보고한 연구는 찾지 못했습니다(→ 열린 질문). 띠 검출에선 33장 라벨로 99% 수준이 보고되었으나 쉬운 조건입니다.
- **엣지+클라우드 캐스케이드 사례**: 범용 계수에서 이를 정량 평가한 논문은 찾지 못했습니다(미확인). "큰 범용 모델 → 작은 미세조정 모델" 증류 경로의 가장 가까운 근거는 RF-DETR 논문(ICLR 2026)입니다. 이 논문은 "VLM 사전지식 + 실시간 검출기"를 목표로 하며, 2x-large 모델이 RF100-VL에서 GroundingDINO(tiny)보다 1.2 AP 높으면서 20배 빠르다고 보고했습니다.

### Q1. 추적에 덜 의존하는 선 통과 계수
- **고전**: Ma & Chan(CVPR 2013, TCSVT 2016)은 기준선의 시간 단면 이미지에서 국소 HOG 특징으로 창별 개수를 회귀하고 정수계획으로 사건 수를 복원했습니다\[29\] — 우리 "시공간 띠" 접근의 원형입니다. 반면 Zhao 외(ECCV 2016)는 "시간 단면의 한계"를 지적하고, 프레임 쌍에서 밀도×속도 맵을 학습해 기존 LOI 방법을 이겼다고 보고했습니다. 이는 **띠가 정보 손실(정지·원근 왜곡)을 가진다는 반대 증거**이므로, 띠 단독보다 "띠 + 프레임 증거 융합"(후보 6)을 권합니다.
- **AI City Challenge Track 1**: 2020년 1~3위(Baidu 0.9389, ENGIE 0.9346, CMU 0.9292)는 모두 같은 3단 구조였습니다. 검출(COCO 사전학습 YOLOv3 다수) → 추적(DeepSORT 다수) → 이동 경로 할당이고, tripwire나 진입/진출 영역은 손으로 그렸습니다. 2021년부터는 온라인(실시간) 알고리즘만 우승 자격을 받았습니다.\[30\] 한계는 손으로 그린 ROI와 카메라별 파라미터 튜닝입니다(ENGIE는 카메라별로 손실을 최적화). **우리의 "기본값 취약성"(오류 6)과 같은 문제를 상위 팀도 수작업으로 풀었다는 뜻**입니다.
- **국소 방식이 이긴 사례**: Ribeiro & Hirata는 띠 기반이 ByteTrack과 정확도는 같으면서(98.56~100%) 속도는 약 3배임을 보였습니다. FCN-rLSTM(ICCV 2017)은 저해상도·저프레임 도시 카메라에서 검출·추적을 피한 밀도 방식으로 MAE를 TRANCOS 5.31 → 4.21, WebCamT 2.74 → 1.53으로 낮췄다고 보고했습니다. 조건: **저해상도·가림이 심해 검출이 불안정할수록 국소/밀도 방식이 유리** — 우리 조건과 일치합니다.

### Q2. 시공간 띠 위 학습형 검출
- 하위 조사 결과, 2018~2026년 띠 위 학습형 모델은 4편(차량 3, 물고기 1)이며, 그중 실제 검출기는 2편(Ribeiro & Hirata, DRIVE-TSI)입니다. 보행자 띠 학습형 검출은 2018년 이후 찾지 못했습니다. **점 라벨로 띠 검출기를 만드는 것은 공개 연구가 거의 없는 빈 칸**입니다.
- 점 라벨 학습: P2PNet(ICCV 2021)은 헝가리안 1:1 매칭으로 점 제안을 학습합니다. P2R(arXiv 2505.21943)은 반지도 학습에서 라벨 5%만으로 완전지도 10%를 넘는다고 보고했습니다 → 새 200개 클립 중 무라벨분을 반지도로 활용할 근거입니다.
- **현실적 기대치(추정)**: 점 1,651개·카메라 4곳이면, 학습에 쓴 카메라에서는 규칙 기반(82~85)을 넘어 90대 초중반이 합리적 목표입니다. 처음 보는 카메라에서는 수%p 하락을 예상합니다(근거: 물고기 사례에서 분포 밖 오차가 커진 비율, 직접 측정 필요).
- **증강**: 띠 시간축 신축(속도 변화 모사), 선 방향 뒤집기(방향 라벨 교환), 밝기·감마·블록 압축 잡음 재인코딩, 차로 띠 잘라 붙이기(밀도 합성). 물고기 연구에서 수직 뒤집기는 도움이 되었으나(48.1 → 43.2), 순진한 수평 뒤집기는 크게 악화시켰습니다(111.5). **시간축 뒤집기는 방향 의미를 깨므로 라벨 교환 없이 쓰면 안 됩니다.**
- **자기지도**: 라벨 없는 띠를 대량 생성(카메라 8곳 × 200클립)해 마스크 복원 사전학습 후 미세조정 — 직접 근거 논문은 미확인이므로 실험 항목으로 둡니다.

### Q3. 작은 원거리·야간
- SAHI(Akyon·Altinuc·Temizel, ICIP 2022)는 VisDrone·xView에서 추론만으로 AP +6.8/+5.1/+5.3%(FCOS/VFNet/TOOD), 슬라이스 미세조정을 더하면 누적 +12.7/+13.4/+14.5%를 보고했습니다. ASAHI(2026)는 SAHI보다 추론시간이 20~25% 줄었다고 보고했습니다. **우리에겐 전체 슬라이싱이 아니라 "기준선 띠 주변만 확대"가 맞습니다** — 계수는 기준선에서만 결정되기 때문입니다.
- 야간 헤드라이트 쌍: 2017년 IEEE 연구는 야간 검출률 81.19%를 보고했고, 2024년 CACIE(Lashkov 외)·ESWA(GMM 쌍 짓기)는 고정 노변 카메라의 야간 교통량 추정을 다룹니다. 이 결과들이 720×480·1 Mbps에서 검증되었다는 원문 근거는 찾지 못했습니다(미확인).
- **우리 조건 해석**: 먼 쪽 헤드라이트는 "차 모양"이 아니라 "움직이는 밝은 점 쌍"이므로, COCO 검출기가 못 잡는 것은 당연합니다. 띠 위에서는 두 개의 평행한 밝은 줄로 나타나므로, **띠 검출기(후보 1)가 야간에 오히려 유리할 가능성**이 큽니다.

### Q4. 촘촘·가림·서다 가다
- OC-SORT·Deep OC-SORT·Hybrid-SORT는 HOTA/IDF1 같은 추적 지표로만 평가됩니다. "선 통과 계수 정확도"를 체계적으로 보고한 동료심사 논문은 드뭅니다. Remote Sensing(2026) 논문은 UA-DETRAC에서 OC-SORT + 궤적 기반 가상 게이트를 썼습니다. 공개 저장소(kapi0okapi/vehicle-counter, 비동료심사)는 같은 검출기에서 추적기 간 계수 오차 차이가 1%p 이내임을 보였습니다.
- 교통공학 가상 루프(후보 5): 띠의 차로별 점유 시계열을 루프 검지기처럼 on/off로 판정합니다. 여기에 "최소 차두 시간(예: 0.5~1초)·최소 점유 길이" 제약을 걸어 쪼개진 덩어리를 병합합니다. 이를 영상에 적용해 정량 비교한 최신 논문은 이번 조사에서 확인하지 못했습니다(미확인).
- 서다 가다 띠 왜곡: DRIVE-TSI의 Cross-Image Object Linking이 "거의 정지한 차의 왜곡"을 명시적으로 다룬 유일한 사례였습니다.\[1\]

### Q5. 새 카메라·화각 변화 적응
- 물고기 사례에서 새 강(분포 밖)으로 가면 오차가 1.3~2.4배로 커졌습니다 → **카메라 단위 교차검증이 필수**입니다.\[31\]
- 적응 경로:
  - (a) 새 카메라 첫 수 분을 클라우드 오토 라벨러로 의사 라벨링한 뒤, 엣지 소형 모델을 수 분간 미세조정.
  - (b) 띠 검출기는 선 위의 1차원 흔적만 보므로 화각 변화에 덜 민감하지만, 원근·차로 폭 정규화가 필요.
  - (c) PTZ 화각 변화는 배경 특징점 매칭 실패율이 급등하는 것으로 감지하고, 차량 궤적을 누적해 차로·진행 방향을 재추정한 뒤 기준선을 다시 배치.
  
  궤적 기반 차로 발견·PTZ 자동 재보정에 대한 최신 정량 연구는 이번 조사 범위에서 확인하지 못했습니다(미확인).

### Q6. 파운데이션 모델의 실용성
- **오토 라벨러로는 "예"**: SAM 3(Carion 외, Meta, arXiv 2511.16719)는 텍스트 "car"만으로 박스·마스크·ID를 줍니다. Meta는 "이미지와 영상의 개념 분할(PCS)에서 기존 시스템보다 정확도가 두 배"라고 보고했습니다(SA-Co/Gold cgF1 54.1, OWLv2 24.6). 코드와 가중치는 github.com/facebookresearch/sam3에 공개되어 있습니다. 단, 주행 영상(BDD100K)에서는 SAM 3 자체 추적(TETA 47.2)보다 SAM 3 검출기 + 단순 추적-by-검출(51.0)이 나았습니다.\[22\] **도로 장면에선 "강한 검출기 + 단순 연결"이 여전히 유리**하다는 뜻으로, 우리 구조에 중요한 신호입니다.
- **실시간 확인기로는 "제한적"**: 848M 파라미터이고, 준실시간은 동시 약 5개 객체 수준입니다(2차 출처). 불일치 사건(전체의 소수)만 ±2초 크롭으로 보내는 경우에 한해 현실적입니다. 사건당 GPU 비용·지연은 미측정이므로 직접 재야 합니다.
- MLLM 계수: Gemini 2.5 Pro가 구조화 프롬프트로 FSC-147 MAE 2.43을 보고했다는 2차 출처가 있으나\[32\] 원문 미확인입니다. 정지 이미지 계수이며 선 통과 사건 판정에는 근거가 없습니다.

### Q7. 엣지 효율·라이선스
- **라이선스가 판단 기준이면 결론은 명확**합니다.
  - Ultralytics YOLO11/YOLO26: AGPL-3.0(또는 유료 Enterprise). 모델에 직접 연결된 코드까지 공개 의무가 생길 수 있습니다.
  - RF-DETR N/S/M/L, D-FINE: Apache-2.0. RF-DETR XL/2XL은 PML 1.0입니다.
  - CoTracker3: CC-BY-NC(비상업)라 상용 불가입니다.
  - SAHI·P2PNet·SAM 3·CountGD++: 라이선스 종류를 이번에 직접 확인하지 못했습니다(미확인 — 도입 전 확인 필수).
- 지연: RF-DETR-N 2.3 ms(T4, TensorRT FP16), YOLO26n CPU ONNX 38.9 ms(제조사 문서).\[12\]\[33\] 지금은 YOLO11s@960이 CPU에서 400 ms인데, 기준선 주변 ROI만 처리하면 입력 면적이 크게 줄어 CPU 실시간에 근접할 여지가 있습니다(직접 측정).
- 캐스케이드·selective prediction: 띠 검출기 피크 신뢰도와 두 경로 일치 여부를 "불확실성"으로 써 클라우드로 보내는 방식이 가장 단순합니다. 신뢰도 보정(온도 스케일링)은 카메라별로 검증 세트에서 맞춥니다.

### Q8. 라벨링·평가·데이터셋
- 평가 지표: 우리 ±1초 헝가리안 사건 매칭은 물고기 계수의 "통과 프레임 기준 사건 집계"와 같은 철학입니다.\[34\] 라벨러 간 일치도 문제(첫 라벨 10~20% 과소 → 재확인 시 구간당 1~2대)는 골든셋에 이중 라벨 + 판정자(adjudication) 단계가 필요하다는 뜻입니다.
- 데이터셋:
  - AI Hub "교통문제 해결을 위한 CCTV 교통 영상(고속도로)": 505시간 영상, 이미지 50만 장(박스 30만, 분할 20만), car/truck/bus 3종. 원래 목적이 "영상VDS(교통량·속도 측정)"로 우리와 같습니다. AI Hub 데이터의 판매 등 상업적 이용은 수행기관과 별도 협의가 필요합니다.\[35\]
  - 경기도 교통 CCTV 데이터: 공공누리 제4유형(출처표시·상업적 이용금지·변경금지).
  - UA-DETRAC(중국 촬영): 추적 GT가 있어 선 통과 사건 GT를 궤적에서 자동 생성할 수 있습니다.
  - AI City Track 1(Iowa DOT): 이동 경로별 계수 GT를 제공합니다(이용 조건은 챌린지 약관 확인 필요).
- 능동 학습: 띠 검출기와 추적 경로가 엇갈린 불일치 사건부터 라벨링하면 라벨 효율이 가장 높을 것으로 봅니다(정량 근거 미확인).

## 4. 실험 계획 (벤치: 검출 캐시 + 세기 규칙 교체, 점 라벨)

| 단계 | 실험 | 구체 절차 | 성공 기준 | 예상 소요 |
|---|---|---|---|---|
| E0 | 기준선 고정 | 현 최고(94.1)를 조건별(낮/밤·밀도·화각)·카메라별로 재채점, 라벨러 이중 확인 구간 표시 | 지표 재현 ±0.3 | 0.5일 |
| E1 | 띠 학습형 점 검출기 | 차로별 띠 생성 → 소형 U-Net/CenterNet(히트맵 + 방향) → leave-one-camera-out 4겹. 증강: 시간 신축·밝기·재압축 | 단독 합계 F1 ≥ 90, 야간 다가오는 방향 재현율 ≥ 80% | 3~5일 |
| E2 | 사건 단위 융합 | E1 사건 ∪ 현 최고 사건, ±1초 매칭, 신뢰도 가중 투표·합집합·교집합 비교 | **합계 F1 94.1 → 96 이상, 대수 오차 −7% → ±3% 이내** | 1일 |
| E3 | 야간 경로 | 기준선 ROI 2배 업샘플 추론 + 띠 위 광원 쌍 채널 | **야간 다가오는 방향 재현율 68% → 85% 이상**, 정밀도 ≥ 95% 유지 | 2~3일 |
| E4 | 오토 라벨 → 미세조정 | 새 200클립에 SAM 3(텍스트 "car/truck/bus")로 의사 박스 → 사람 점과 대조해 의사 라벨 재현율 측정 → RF-DETR-S 미세조정 → 캐시 재생성 | 의사 라벨 재현율 ≥ 95%; 검출 경로 단독 F1 +2 이상; 라이선스 Apache 경로 확보 | 4~6일 |
| E5 | 서다 가다 규칙 | E1 출력에 가상 루프 on/off + 최소 차두 제약 | 서다 가다 구간 정밀도·재현율 각 +3%p | 1일 |
| E6 | 추적기 교체(저비용 확인) | 캐시 위 OC-SORT/Hybrid-SORT | 촘촘 낮 재현율 +2%p 이상이면 채택, 아니면 종료 | 0.5일 |
| E7 | 캐스케이드 비용 | E2 불일치 사건만 CountGD++/SAM 3 확인, 호출률·지연·정확도 기록 | 호출률 ≤ 10% 사건, 확인 후 F1 +0.5 이상 | 2일 |

**두 번째 도메인 이식 시험(제안)**: MOT20(또는 MOT17) 보행자 시퀀스를 씁니다.
- 준비: GT 궤적에서 임의 기준선 통과 사건을 자동 생성합니다(사람 라벨 불필요).
- 방법: 차량용과 **똑같은 코드**(띠 생성·점 검출기·융합·채점)를 쓰고, 학습 라벨만 해당 시퀀스 일부의 통과 점으로 바꿉니다.
- 측정: (a) 차량 가중치로 0-shot, (b) 점 50/200/500개로 미세조정했을 때의 F1 곡선.
- 성공 기준: 점 200개 이하로 F1 90 이상.

보조 비교로 CountVid(텍스트 "person")의 영상 고유 수와 대조합니다. 세 번째 후보는 LOI 프로토콜이 같은 Caltech Fish Counting입니다.

## 5. 범용 솔루션 구조 제안 — "Count anything, Track everything"

| 계층 | 역할 | 지금(차량) | 다른 도메인으로 옮길 때 바뀌는 것 |
|---|---|---|---|
| ① 대상 지정 | 무엇을 셀지 | 텍스트 "car/truck/bus" + 카메라별 예시 박스 1~3개 + 부정 예시(자막·그림자) | 텍스트·예시만 교체("person", "salmon", "cell") |
| ② 클라우드 범용 모델 | 오토 라벨러·불일치 확인기 | SAM 3 / CountGD++ / CountVid | 그대로 |
| ③ 엣지 모델 | 상시 처리 | 띠 점 검출기(주) + RF-DETR-N/S 검출기(보조), Apache | ②의 의사 라벨로 재증류 |
| ④ 계수 규칙 | 사건 정의 | 기준선(차로별)·방향·띠 피크·가상 루프·최소 차두 | 선/영역 기하와 최소 간격 파라미터만 교체 |
| ⑤ 융합·캐스케이드 | 신뢰도·불일치 라우팅 | ±1초 사건 매칭, 불일치만 ②로 | 허용 시간창만 교체 |
| ⑥ 적응 | 새 카메라·화각 | 화각 변화 감지 → 궤적으로 차로 재추정 → 기준선 재배치 → 의사 라벨 재학습 | 동일 절차 |
| ⑦ 평가 | 사건 단위 F1·창별 대수 오차·지연 | 점 라벨 골든셋(이중 라벨) | 동일 지표 |

```
[영상] ─┬─> [기준선 띠 생성] ──> [띠 점 검출기(엣지)] ──┐
        └─> [ROI 확대 검출(엣지)] ─> [단순 추적] ─> [선 통과] ─┤
                                                         ▼
                                     [사건 융합 ±Δt, 신뢰도]
                                       │일치            │불일치/저신뢰
                                       ▼                ▼
                                   [집계·방향별]   [클라우드: SAM 3 / CountGD++ 확인]
                                                        │
                              [의사 라벨 축적] ──> [엣지 모델 재학습(적응)]
```

## 6. 읽을거리 (분야별)

**범용·오픈 어휘 계수**
- CountGD (NeurIPS 2024) — 텍스트·예시 다중 모달 계수. https://arxiv.org/abs/2407.04619
- CountGD++ (arXiv 2512.23351, CVPR 2026) — 부정 프롬프트·의사 예시·외부 예시. https://arxiv.org/abs/2512.23351
- CountVid (AAAI 2026) — 영상 고유 개체 계수 + VideoCount. https://arxiv.org/abs/2506.15368 \[36\]
- GeCo (NeurIPS 2024) — 검출 기반 저샷 계수가 밀도 기반을 넘음. https://arxiv.org/abs/2409.18686 \[37\]
- SAM 3 (2025) — 개념 프롬프트 검출·분할·추적. https://arxiv.org/abs/2511.16719

**선 통과(LOI)·영상 개체 계수**
- Ma & Chan, Crossing the Line (CVPR 2013) — 시간 단면 + 정수계획. https://openaccess.thecvf.com/content_cvpr_2013/papers/Ma_Crossing_the_Line_2013_CVPR_paper.pdf
- Zhao 외, Crossing-line Crowd Counting (ECCV 2016) — 밀도 × 속도 맵. https://www.ee.cuhk.edu.hk/~xgwang/papers/ZhaoLZWeccv16.pdf
- DRNet / DR.VIC (CVPR 2022) — inflow 분해로 영상 개체 계수. https://arxiv.org/abs/2203.12335
- 약지도 VIC (arXiv 2312.05923) — inflow/outflow 라벨만으로 학습. https://arxiv.org/abs/2312.05923
- 군집·변위 사전 VIC (arXiv 2601.01192) — 일대다 매칭. https://arxiv.org/abs/2601.01192

**시공간 띠·교통 계수**
- Ribeiro & Hirata, YOLO + Visual Rhythm (SIBGRAPI 2023) — 띠 위 YOLO로 99% 수준. https://arxiv.org/abs/2501.04534
- DRIVE-TSI (IEEE 2021) — TSI 검출 + 정지 차 연결. https://ieeexplore.ieee.org/document/9637752/
- FCN-rLSTM (ICCV 2017) — 저품질 도시 카메라 밀도+LSTM 계수. https://arxiv.org/abs/1707.09476
- The 4th AI City Challenge (CVPRW 2020) — Track 1 상위 팀 구조 요약. https://arxiv.org/abs/2004.14619

**생태·과학 영상**
- Caltech Fish Counting (ECCV 2022) — 소나 LOI 계수·도메인 일반화. https://arxiv.org/abs/2207.09295 \[31\]
- Counting Fish with Temporal Representations of Sonar Video (2025) — 에코그램 직접 회귀. https://arxiv.org/abs/2502.05129 \[5\]
- CFC26 (2026) — 9개 강으로 확장한 배치용 벤치마크. https://arxiv.org/html/2610.22549 \[34\]

**점 라벨·작은 물체·엣지**
- P2PNet (ICCV 2021) — 순수 점 기반 계수·위치. https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- P2R (arXiv 2505.21943) — 반지도 점 계수. https://arxiv.org/abs/2505.21943
- SAHI (ICIP 2022) — 슬라이싱 추론. https://arxiv.org/abs/2202.06934
- RF-DETR (ICLR 2026) — Apache 실시간 DETR. https://arxiv.org/abs/2511.09554

**추적**
- OC-SORT (CVPR 2023) https://arxiv.org/abs/2203.14360 · Deep OC-SORT https://arxiv.org/abs/2302.11813 · Hybrid-SORT https://arxiv.org/abs/2308.00783

## 7. 열린 질문 (연구로 답이 없고 직접 실험해야 할 것)
1. 띠 점 검출기가 **서다 가다**에서 "통과 순간"을 어떻게 정의해야 사람 라벨과 일치하는가(띠에서 정지 차는 시간축으로 긴 줄 — 앞 범퍼 기준? 중앙 기준?).
2. 처음 보는 카메라에서 F1 90을 넘기기 위한 **최소 점 라벨 수**(10/50/200개?) — 범용 계수 문헌에 사건 단위 기준의 답이 없습니다.
3. SAM 3·CountGD++가 **1 Mbps·720×480·야간**에서 의사 라벨로 쓸 만큼 정확한가(공개 벤치는 고화질 위주).
4. 클라우드 확인기의 **사건당 비용·지연**과, 불일치 사건 호출률이 실제로 10% 이하로 유지되는가.
5. 기준선 정의 민감도(오류 5)를 줄이려면 선 대신 **얇은 영역(띠 폭 수 px)** 통과로 정의하는 것이 더 안정적인가.
6. PTZ 화각 변화 후 **자동 기준선 재배치**가 사람이 그은 선 대비 F1을 얼마나 잃는가.
7. 사람 라벨 자체의 불확실성(첫 라벨 10~20% 과소)을 감안할 때, 96 이상 F1이 라벨 잡음 상한에 이미 가까운지.

## 8. 주의사항 (Caveats)
- 띠 검출의 높은 정확도(97.95~99.15%)는 상단 시점·주간·단방향 조건이며 우리 조건으로의 전이는 검증되지 않았습니다.
- 추적기 간 계수 오차 비교(1%p 이내)와 YOLO26 미세조정 효과(9.7% → 5.7% → 3.3%)는 동료심사 없는 공개 저장소 결과입니다.\[18\]
- YOLO26의 "CPU 최대 43% 빠름"과 RF-DETR의 우위 주장은 각 제조사(Ultralytics, Roboflow)가 자체 측정한 것입니다. 지연 수치는 측정 환경마다 크게 달라지며, 대부분 T4 TensorRT 기준이고 CPU·Jetson 지연은 거의 공개되지 않았습니다.
- CountGD FSC-147 MAE 5.74, Gemini 2.5 Pro MAE 2.43, SAM 3 준실시간 객체 수는 2차 출처로만 확인했습니다.
- DRIVE-TSI와 TM-Net은 초록만 확인했으며, 야간·정체 세부 수치는 미확인입니다.
- 예상 효과(높음/중간/낮음)와 성공 기준 수치는 근거 기반 추정이며 측정값이 아닙니다.

## 출처

1. [Efficient Vehicle Counting Based On Time-Spatial Images By Neural Networks](https://ieeexplore.ieee.org/document/9637752/)
2. <https://arxiv.org/pdf/2501.04534>
3. [IEEE Access (Jan 2021)](https://doaj.org/article/450760c665e44accb6b1f0a5619e90d8)
4. [Counting Fish with Temporal Representations of Sonar Video](https://arxiv.org/abs/2502.05129)
5. [Counting Fish with Temporal Representations of Sonar Video · Pith Review](https://pith.science/paper/2502.05129)
6. [Paper page - Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery](https://huggingface.co/papers/2604.19233)
7. [Monocular 3D Object Detection using Multi-Stage Approaches with Attention and Slicing aided hyper inference](https://arxiv.org/pdf/2212.11804)
8. [GitHub - roboflow/rf-detr: RF-DETR is a real-time object detection and segmentation model architecture developed by Roboflow, SOTA on COCO, designed for fine-tuning. \[ICLR 2026\] · GitHub](https://github.com/roboflow/rf-detr)
9. [RF-DETR vs. Alternatives: Benchmarks and Deployment Compared](https://roboflow.com/blog/rf-detr-vs-alternatives)
10. [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/pdf/2511.09554)
11. [replicate.com](https://replicate.com/ultralytics/yolo26)
12. [YOLO26 vs YOLOv8 Comparison](https://docs.ultralytics.com/compare/yolo26-vs-yolov8)
13. [Open-World Object Counting in Videos](https://arxiv.org/html/2506.15368v1)
14. [DR.VIC: Decomposition and Reasoning for Video Individual Counting Tao Han1†,](https://arxiv.org/pdf/2203.12335)
15. [Weakly Supervised Video Individual Counting](https://arxiv.org/html/2312.05923v1)
16. <https://arxiv.org/pdf/2502.05129v1>
17. [Nighttime vehicle detection and classification via headlights trajectories matching](https://ieeexplore.ieee.org/document/8030869/)
18. [GitHub - kapi0okapi/vehicle-counter: YOLO26 and five multi-object trackers count each vehicle once per line and direction - 4.7 % counting error on traffic cameras it never saw. · GitHub](https://github.com/kapi0okapi/vehicle-counter)
19. [Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking](https://arxiv.org/pdf/2308.00783)
20. [Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification](https://arxiv.org/pdf/2302.11813)
21. [SAM 3: Segment Anything with Concepts](https://www.alphaxiv.org/overview/2511.16719v1)
22. [SAM 3: Segment Anything with Concepts](https://arxiv.org/pdf/2511.16719)
23. [CountGD: Multi-Modal Open-World Counting \[Quick Review\]](https://liner.com/review/countgd-multimodal-openworld-counting)
24. [CountVid: Open-World Object Counting in Videos](https://www.robots.ox.ac.uk/~vgg/research/countvid/)
25. [Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors](https://arxiv.org/html/2601.01192)
26. [Open-World Object Counting in Videos Niki Amini-Naieni, Andrew Zisserman](https://www.robots.ox.ac.uk/~vgg/publications/2026/AminiNaieni26/amininaieni26.pdf)
27. [SAM 3: Segment Anything with Concepts — Lacuna](https://lacuna.tiptreesystems.com/work/sam-3-segment-anything-with-concepts/wrk_418d97bccbb7ba5aed9d5ea16b11f5d8)
28. [Meta AI Releases Segment Anything Model 3 (SAM 3) for Promptable Concept Segmentation in Images and Videos - MarkTechPost](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/)
29. [Crossing the Line: Crowd Counting by Integer Programming with Local Features](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ma_Crossing_the_Line_2013_CVPR_paper.pdf)
30. [\[2104.12233\] The 5th AI City Challenge](https://ar5iv.labs.arxiv.org/html/2104.12233)
31. [\[2207.09295\] The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting](https://arxiv.org/abs/2207.09295)
32. [Multi-Modal Open-World Counting](https://www.emergentmind.com/topics/multi-modal-open-world-counting)
33. [RF-DETR — Real-Time Object Detection & Segmentation by Roboflow](https://rfdetr.roboflow.com/)
34. [CFC26: Building Evaluations for Deployment in Sonar-Based Fish Counting](https://arxiv.org/html/2610.22549)
35. [이용정책 - AI 허브소개 - AI-Hub](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)
36. [\[2506.15368\] Open-World Object Counting in Videos](https://arxiv.org/abs/2506.15368)
37. [NeurIPS Poster A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation](https://neurips.cc/virtual/2024/poster/93724)
