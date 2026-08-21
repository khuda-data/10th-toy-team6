# 10th-toy-team6

6조 토이프로젝트 — Steam 사용자 맞춤형 Value Score 기반 게임 추천 시스템

## 프로젝트 구조

```
10th-toy-team6/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example            # STEAM_API_KEY= 형태로 키만 비워둔 예시. 복사해서 .env로 사용
├── data/
│   ├── raw/                 # Steam API로 긁어온 원본 (git에는 커밋하지 않음)
│   └── processed/           # 전처리 끝난 데이터 (기존 "전처리 끝난 데이터" 폴더)
│       ├── steam_top500_games_classified.csv
│       ├── steam_user_games_classified.csv
│       └── user_genre_weights.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── legacy/               # 가중치 부여 기반 추천 실험 노트북
│       ├── weighted_lightgbm.ipynb
│       ├── weighted_random_forest.ipynb
│       └── weighted_xgboost.ipynb
└── docs/
    ├── 최종모델.md                     # 추천 시스템 설계 문서
    └── PROJECT_QUALITY.md              # 프로젝트 구조/보안 이슈 점검 및 개선 이력
```

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env   # STEAM_API_KEY 채워넣기 (notebooks/legacy에서 사용)
```

`notebooks/legacy/`의 노트북들은 `data/processed/`에 있는 CSV를 상대경로로 읽습니다.

## 참고

`docs/PROJECT_QUALITY.md`에 구조 정리 배경과 처리한 이슈(Steam API Key 하드코딩 노출 등)가 정리되어 있습니다.
