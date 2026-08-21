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
│   └── processed/           # 전처리 끝난 데이터 + feature (game_features.csv, user_features.csv 포함)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── legacy/               # 예전 분류기 기반 추천 실험 (참고용, src/recommenders/classifier_based.py로 통합됨)
│       ├── weighted_lightgbm.ipynb
│       ├── weighted_random_forest.ipynb
│       └── weighted_xgboost.ipynb
├── src/
│   ├── features.py                   # 게임/유저 feature 생성
│   ├── evaluate.py                    # 공통 평가 지표 (Leave-One-Out, Precision@K, NDCG@K 등)
│   ├── run_comparison.py              # 추천 모델 6종 비교 실행 진입점
│   └── recommenders/
│       ├── content_based.py           # Popularity, ContentBased
│       ├── collaborative.py           # UserCF, ItemCF, ALS, Hybrid
│       └── classifier_based.py        # LightGBM/RandomForest/XGBoost 공통 모듈 (legacy 노트북 3개 통합)
├── reports/                            # 비교 결과 (표, 그래프)
│   └── README.md                       # 추천 모델별 성능 비교 결과 및 해석
└── docs/
    ├── 최종모델.md                     # 추천 시스템 설계 문서
    └── PROJECT_QUALITY.md              # 프로젝트 구조/보안 이슈 점검 및 개선 이력
```

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env   # STEAM_API_KEY 채워넣기 (legacy classifier_based.py에서 사용)

cd src
python features.py         # data/processed/game_features.csv, user_features.csv 생성
python run_comparison.py   # Popularity/ContentBased/UserCF/ItemCF/ALS/Hybrid 비교 -> reports/
```

분류기 기반(레거시) 추천을 유저 1명에 대해 돌려보고 싶다면:

```bash
cd src
python -m recommenders.classifier_based --steam-id <STEAM_ID> --model lightgbm
```

## 추천 시스템 비교 결과

`reports/README.md`에 전체 878명 유저 기준 Leave-One-Out 평가 결과와 해석이 정리되어 있습니다. 요약하면 NDCG 기준 UserCF(협업 필터링)가 가장 좋았고, 장르 라벨만 쓰는 ContentBased가 가장 낮았습니다.

## 프로젝트 히스토리 / 알려진 이슈

`docs/PROJECT_QUALITY.md`에 구조 정리 배경과 처리한 이슈(Steam API Key 하드코딩 노출 등)가 정리되어 있습니다.
