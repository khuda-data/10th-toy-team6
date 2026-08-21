# 프로젝트 정리 이력

`restructure-project` 브랜치에서 아래 구조 개편을 실제로 적용했습니다. 이 문서는 "왜 혼란스러웠는지"와 "무엇을 어떻게 고쳤는지"를 정리한 기록입니다.

## 🚨 지금 바로 처리해야 하는 것: Steam API Key 유출

`가중치 부여 + lightgbm.ipynb`, `random forest.ipynb`, `xgboosting.ipynb` 3개 파일 모두에

```python
STEAM_API_KEY = "B1FC36C7D790B5F17DCD8E33F5C33DF2"
```

가 평문으로 박혀 있고, git 커밋 히스토리에도 이미 남아있는 걸 확인했습니다 (`git log -p`로 12회 검색됨). 저장소가 `github.com/khuda-data/10th-toy-team6`로 원격에 올라가 있으니, 리포가 public이면 이미 노출된 키입니다.

**처리 상태**
1. ⬜ **[Steam 개발자 페이지](https://steamcommunity.com/dev/apikey)에서 이 키를 즉시 재발급(폐기 후 재발급)** — 이건 코드로 대신 해드릴 수 없어서 직접 처리해주셔야 합니다.
2. ✅ 코드에서 하드코딩된 키를 제거하고 `os.getenv("STEAM_API_KEY")` + `.env` + `python-dotenv`로 교체 (`notebooks/legacy/*.ipynb`)
3. ✅ `.env`를 `.gitignore`에 추가
4. ⬜ git 히스토리에서 완전히 지우려면 `git filter-repo`나 BFG Repo-Cleaner가 필요 (팀 전체 재-clone 필요) — 이번엔 "재발급만, 히스토리는 그대로" 두는 쪽으로 결정. 필요해지면 팀원들과 시간 맞춰서 진행하세요.

## 폴더 구조가 혼란스러웠던 이유

정리 전 구조:
```
10th-toy-team6/
├── 01_data_collection.ipynb
├── 가중치 부여 + lightgbm.ipynb
├── 가중치 부여 + random forest.ipynb
├── 가중치 부여 + xgboosting.ipynb
├── 최종모델.md
├── 전처리 끝난 데이터/
│   ├── steam_top500_games_classified.csv
│   ├── steam_user_games_classified.csv
│   └── user_genre_weights.csv
└── .ipynb_checkpoints/   ← git에 커밋되어 있음 (지워야 함)
```

문제점:
- **소스 코드, 데이터, 문서가 한 폴더에 뒤섞여 있음** → 뭐가 최신이고 뭐가 실행 가능한지 파악이 안 됨
- **"가중치 부여 + X.ipynb" 3개가 사실상 동일 코드의 복붙** (분류기 15줄 정도만 다름) → 하나 고치면 나머지 2개도 일일이 고쳐야 함
- **`.ipynb_checkpoints/`가 git에 커밋됨** (자동 생성되는 캐시 폴더라 커밋하면 안 됨)
- **경로가 `C:\Users\User\...`로 하드코딩** → 작성자 본인 컴퓨터에서만 실행 가능, 팀원은 못 돌림
- **파일명 불일치**: 노트북은 `steam_user_games_merged.csv`를 읽는데 실제 데이터는 `steam_user_games_classified.csv` → 지금 상태로는 노트북이 아예 안 돌아감
- `.gitignore`가 없음 → 캐시/중간산출물이 계속 커밋될 위험

## 적용한 구조

```
10th-toy-team6/
├── README.md                      # 프로젝트 개요, 실행 방법, 팀원
├── requirements.txt
├── .gitignore
├── .env.example                   # STEAM_API_KEY= 형태로 키만 비워둔 예시
├── data/
│   ├── raw/                       # API로 긁어온 원본 (.gitkeep만 커밋, 실제 원본은 gitignore)
│   └── processed/                 # 전처리 끝난 데이터 (기존 "전처리 끝난 데이터" 폴더)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── legacy/                    # 예전 "가중치 부여 + X.ipynb" 3개 (API 키/경로만 정리, 파일명은 영문으로)
└── docs/
    ├── 최종모델.md
    └── PROJECT_QUALITY.md         # 이 문서
```

핵심은 **데이터 / 노트북 / 문서를 분리**하고, 원래 있던 파일들을 손대지 않은 채로(로직 변경 없이) 제자리를 찾아준 것입니다. 노트북 3개가 사실상 복붙 코드라 공통 모듈로 합치면 좋겠지만, 그건 로직을 새로 작성하는 일이라 이번 정리 범위에서는 제외했습니다.

## 적용한 .gitignore

```gitignore
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
.DS_Store
*.pkl
.venv/
venv/
data/raw/*
!data/raw/.gitkeep
```

`.ipynb_checkpoints/`는 git 추적에서 제거했습니다 (`git rm -r --cached`). `data/processed/`의 산출물은 팀 공유용이라 그대로 커밋했지만, 원본 대용량 raw 데이터가 생기면 git이 아니라 별도 공유 방식(Drive 등)을 쓰는 걸 권합니다.

## main과 merge 이후 업데이트

`main`에 팀원이 올린 새 전처리 파이프라인(`02_data_cleaning.ipynb`, `03_data_preprocessing.ipynb`, `data/raw/original_data.csv`, `data/processed/*.csv` 8종)을 `restructure-project`로 merge하면서, 예전 `가중치 부여 + lightgbm/random forest/xgboosting.ipynb` 3개와 그 노트북들이 쓰던 구 데이터(`steam_top500_games_classified.csv`, `steam_user_games_classified.csv`, `user_genre_weights.csv`)를 완전히 제거했습니다. 팀원이 이미 이 데이터를 새 파이프라인으로 대체했기 때문에, 예전 노트북은 실행할 데이터가 없어져서 어차피 못 돌아가는 상태였습니다.

**Steam API Key 참고**: 위에서 언급한 키는 이제 제거된 노트북 3개에만 있던 것이라 현재 작업 트리에는 없지만, git 히스토리에는 여전히 남아있습니다. **재발급은 여전히 처리해주셔야 합니다.**

## 그 외 적용한 개선

- `최종모델.md`를 `docs/`로 이동했습니다.
- 노트북 파일명을 `weighted_lightgbm.ipynb`처럼 영문 스네이크 케이스로 바꿨습니다 (공백/`+` 제거).
- 루트에 README.md를 추가해서 구조를 안내합니다.
- 남은 일: 위 "지금 바로 처리해야 하는 것"의 API 키 재발급은 코드로 할 수 없는 부분이라 직접 처리해주셔야 합니다.
