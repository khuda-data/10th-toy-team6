# 10th-toy-team6

6조 토이프로젝트 — Steam 사용자 맞춤형 Value Score 기반 게임 추천 시스템

## 프로젝트 구조

```
10th-toy-team6/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example         # STEAM_API_KEY= 형태로 키만 비워둔 예시. 복사해서 .env로 사용
├── data/                 # 데이터
│   ├── raw/              # 원본 데이터
│   │   └── original_data.csv
│   └── processed/         # 전처리/가공 끝난 데이터
│       ├── cleaned_data.csv
│       ├── game_features_full.csv
│       ├── user_genre_preference_raw.csv
│       ├── user_genre_preference_normalized.csv
│       ├── user_tag_preference_top30_raw.csv
│       ├── user_tag_preference_top30_normalized.csv
│       ├── user_tag_preference_top50_raw.csv
│       ├── user_tag_preference_top50_normalized.csv
│       └── user_profile_features.csv
├── notebooks/            # 실험/분석용 Jupyter Notebook
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_data_preprocessing.ipynb
├── src/                  # 재사용할 실제 Python 코드
├── results/              # 실행 결과
└── docs/                 # 문서
    ├── 최종모델.md         # 추천 시스템 설계 문서
    └── PROJECT_QUALITY.md    # 프로젝트 구조/보안 이슈 점검 및 개선 이력
```

## 파일 흐름

```
data/raw              (원본 데이터 수집)
   │
   ▼
data/processed         (정제·전처리 끝난 데이터, feature)
   │
   ▼
notebooks/              (01 수집 → 02 정제 → 03 전처리, 여기서 자유롭게 탐색/실험)
   │  ── 검증된 로직만 뽑아서 ──▶ src/  (재사용 가능한 Python 모듈로 정리)
   ▼
results/                (노트북·src 실행 결과: 추천 목록, 성능 비교표, 그래프 등 저장)
   │
   ▼
docs/                   (실험 결과와 설계 내용을 문서로 정리)
```

즉 `data`에서 시작해서 `notebooks`에서 실험하고(01_data_collection → 02_data_cleaning → 03_data_preprocessing 순서), 반복해서 쓸 코드는 `src`로 옮기고, 실행 결과는 `results`에 쌓고, 최종적으로 `docs`에 정리하는 흐름입니다.

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env   # STEAM_API_KEY 채워넣기
```

## 참고

`docs/PROJECT_QUALITY.md`에 구조 정리 배경과 처리한 이슈(Steam API Key 하드코딩 노출 등)가 정리되어 있습니다.
