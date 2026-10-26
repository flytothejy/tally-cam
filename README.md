# tally-cam

공공 CCTV(ITS 국가교통정보센터) 위에 차량 카운팅을 얹는 저사양 엣지 추론 테스트베드.

## 구조
- `scripts/snapshot.py` — ITS API 1회 호출로 CCTV 목록을 받아 `config.json`의 카메라만 한 프레임씩 저장
- `.github/workflows/snapshot.yml` — 하루 3회(KST 07:17 / 13:17 / 20:17) 스냅샷 수집 후 커밋
- `index.html` — GitHub Pages로 최신 스냅샷과 최근 기록을 보여주는 페이지
- `data/` — 수집 결과 (카메라별 최근 24장 유지)

## 보안
ITS 스트림 서명 URL에는 API 키가 포함되어 있다. 스트림 URL은 절대 출력·커밋·공개하지 않는다.
키는 로컬에선 `.env`(gitignore), Actions에선 Secret `ITS_API_KEY`로만 다룬다.

## 로컬 실행
```
echo "ITS_API_KEY=발급키" > .env
python3 scripts/snapshot.py
```

## API 한도
개발용 키 월 100건 → 하루 3회(~93건). 운영 키 승인 후 cron 주기를 늘린다.

영상 출처: 국토교통부 ITS 국가교통정보센터 (공공누리 제1유형)
