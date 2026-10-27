# food-kcal

음식 사진을 올리면 음식 종류를 인식하고 칼로리와 영양성분을 표시하는 프로그램입니다.

- 문서: [Wiki](https://github.com/bio-hyeok/food-kcal/wiki)

## 폴더 구조

```
food-kcal/
├─ server/              # API 서버
│  ├─ db/               # DB 스키마 (마이그레이션)
│  └─ inference/        # 추론 서버 (ONNX 분류 모델, 사진 → 음식 후보)
├─ app/                 # 클라이언트
├─ ml/                  # 학습·추론
│  ├─ scripts/          # 전처리, 학습, 평가 스크립트
│  ├─ configs/          # 학습 설정
│  └─ label_map.csv     # 데이터셋 라벨 ↔ food_id 매핑
├─ data/                # (Git 제외) 다운로드한 데이터
├─ models/              # (Git 제외) 학습된 모델 파일
└─ README.md
```

`data/`와 `models/`의 내용물은 Git에 올리지 않습니다. 폴더만 `.gitkeep`으로 유지합니다.

## 브랜치 규칙

| 브랜치 | 역할 |
| --- | --- |
| `main` | 배포용. `develop`에서 확인된 것만 merge |
| `develop` | 통합·확인용 |
| `feature/*` | 작업용. `develop`에서 생성하고 `develop`으로 merge |
