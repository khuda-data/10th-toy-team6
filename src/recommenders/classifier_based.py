# -*- coding: utf-8 -*-
"""
분류기 기반 개인 맞춤 추천 (LightGBM / RandomForest / XGBoost 공통 모듈).

기존에는 노트북 3개(가중치 부여 + lightgbm/random forest/xgboosting.ipynb)가
분류기 15줄 정도만 다르고 나머지 로직은 100% 동일했다. 이 모듈은 그 공통 로직을
하나로 합치고, 분류기 종류만 `model_type` 인자로 선택하게 만든 버전이다.

주의 (원래 구조의 한계, docs/PROJECT_QUALITY.md 참고):
  이 방식은 Steam ID 1명 단위로 즉석에서 모델을 새로 학습한다.
  - positive = TOP500 중 해당 유저가 플레이한 게임
  - negative = TOP500 중 해당 유저가 안 한 게임
  학습 샘플이 유저 1명당 최대 수십 개뿐이라 일반화 성능은 제한적이다.
  전체 유저에 대해 배치로 비교하려면 src/evaluate.py의 Leave-One-Out 프레임을
  쓰는 협업 필터링 계열(src/recommenders/collaborative.py)을 참고할 것.

사용 예:
    from recommenders.classifier_based import run_for_user
    run_for_user(steam_id="7656119...", model_type="lightgbm")
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/ 를 import 경로에 추가
from features import GENRES, determine_genres, normalize_text  # noqa: E402

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_CSV = REPO_ROOT / "data" / "processed" / "steam_user_games_classified.csv"
TOP500_CSV = REPO_ROOT / "data" / "processed" / "steam_top500_games_classified.csv"
REPORTS_DIR = REPO_ROOT / "reports"

STEAM_API_KEY = os.getenv("STEAM_API_KEY", "")

COL_STEAM_ID = "steamid"
COL_GAME = "game_name"
COL_PLAYTIME = "playtime_hours"
COL_GENRE = "genre"
COL_TAGS = "tags"

MODEL_REGISTRY = {}


def _register_models():
    """지원 라이브러리가 설치된 경우에만 레지스트리에 등록 (선택적 의존성)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        MODEL_REGISTRY["random_forest"] = lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        MODEL_REGISTRY["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=10, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", objective="binary", random_state=42,
            n_jobs=-1, verbosity=-1,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        MODEL_REGISTRY["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass


_register_models()


# ------------------------------------------------------------------
# 데이터 로드 / 전처리
# ------------------------------------------------------------------

def load_data():
    user_df = pd.read_csv(USER_CSV, dtype={COL_STEAM_ID: str}, encoding="utf-8-sig")
    top500_df = pd.read_csv(TOP500_CSV, dtype={"appid": str}, encoding="cp949")
    return user_df, top500_df


def get_user_games(user_df, steam_id):
    steam_id = str(steam_id).strip()
    df = user_df[user_df[COL_STEAM_ID].astype(str).str.strip() == steam_id].copy()
    if df.empty:
        raise ValueError(f"Steam ID '{steam_id}'의 데이터를 찾을 수 없습니다.")
    df[COL_PLAYTIME] = pd.to_numeric(df[COL_PLAYTIME], errors="coerce")
    df = df.dropna(subset=[COL_PLAYTIME])
    df = df[df[COL_PLAYTIME] > 0].copy()
    df = df.groupby([COL_GAME, COL_GENRE, COL_TAGS], dropna=False, as_index=False)[COL_PLAYTIME].sum()
    return df


def get_steam_library(steam_id):
    """Steam Web API로 유저의 전체 보유 게임을 조회 (추천에서 제외하기 위함).
    STEAM_API_KEY가 .env에 설정되어 있어야 한다."""
    if not STEAM_API_KEY:
        print("STEAM_API_KEY가 설정되지 않았습니다 (.env 확인). 보유 게임 필터링을 건너뜁니다.")
        return set()

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {"key": STEAM_API_KEY, "steamid": steam_id, "include_appinfo": 1, "include_played_free_games": 1}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Steam API 요청 실패:", e)
        return None

    games = data.get("response", {}).get("games", [])
    return {normalize_text(g.get("name", "")) for g in games if g.get("name")}


def calculate_weights(user_games):
    df = user_games.copy()
    playtime = df[COL_PLAYTIME].astype(float).to_numpy()
    log_playtime = np.log1p(playtime)
    total = log_playtime.sum()
    df["weight"] = (1 / len(df)) if total == 0 else (log_playtime / total)
    return df


def build_user_preference(weighted_games):
    preference = {g: 0.0 for g in GENRES}
    for _, row in weighted_games.iterrows():
        genres = determine_genres(row[COL_GENRE], row[COL_TAGS])
        w = float(row["weight"])
        for g in genres:
            if g in preference:
                preference[g] += w
    total = sum(preference.values())
    if total > 0:
        preference = {k: v / total for k, v in preference.items()}
    return preference


def create_game_features(top500_df):
    rows = []
    for _, row in top500_df.iterrows():
        vector = {g: int(g in determine_genres(row.get(COL_GENRE, ""), row.get(COL_TAGS, ""))) for g in GENRES}
        result = {"appid": str(row["appid"]), "game_name": row[COL_GAME]}
        result.update({f"game_{g}": vector[g] for g in GENRES})
        rating = pd.to_numeric(row.get("rating", np.nan), errors="coerce")
        result["rating"] = 0.0 if pd.isna(rating) else float(rating)
        rows.append(result)
    return pd.DataFrame(rows)


def create_training_data(user_games, game_features):
    played = set(user_games[COL_GAME].astype(str).str.lower().str.strip())
    X, y = [], []
    for _, game in game_features.iterrows():
        name = normalize_text(game["game_name"])
        feature = [game[f"game_{g}"] for g in GENRES] + [game["rating"]]
        X.append(feature)
        y.append(1 if name in played else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


# ------------------------------------------------------------------
# 학습 / 평가
# ------------------------------------------------------------------

def train_classifier(X, y, model_type):
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"'{model_type}'는 지원하지 않거나 라이브러리가 설치되지 않았습니다. 사용 가능: {available}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None,
    )
    model = MODEL_REGISTRY[model_type]()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["pr_auc"] = average_precision_score(y_test, y_prob)
    except Exception:
        metrics["roc_auc"] = metrics["pr_auc"] = 0.0

    print(f"=== {model_type} 평가 ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, metrics


def recommend_games(model, game_features, user_preference, steam_library, top_n=20):
    df = game_features.copy()
    X = np.array([[g[f"game_{gn}"] for gn in GENRES] + [g["rating"]] for _, g in df.iterrows()], dtype=float)
    df["probability"] = model.predict_proba(X)[:, 1]
    df["genre_preference_score"] = [
        sum(g[f"game_{gn}"] * user_preference[gn] for gn in GENRES) for _, g in df.iterrows()
    ]
    df["final_score"] = df["probability"]
    df["game_name_lower"] = df["game_name"].astype(str).str.lower().str.strip()

    if steam_library:
        df = df[~df["game_name_lower"].isin(steam_library)].copy()

    df = df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def run_for_user(steam_id, model_type="lightgbm", top_n=20, save=True):
    """단일 유저에 대해 지정한 분류기로 추천을 생성한다."""
    user_df, top500_df = load_data()
    user_games = get_user_games(user_df, steam_id)
    steam_library = get_steam_library(steam_id) or set()

    weighted_games = calculate_weights(user_games)
    user_preference = build_user_preference(weighted_games)
    game_features = create_game_features(top500_df)

    X, y = create_training_data(user_games, game_features)
    if len(X) < 10 or len(np.unique(y)) < 2:
        raise ValueError("학습 데이터가 부족합니다 (positive/negative가 둘 다 충분히 필요).")

    model, metrics = train_classifier(X, y, model_type)
    recommendations = recommend_games(model, game_features, user_preference, steam_library, top_n=top_n)

    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        out_path = REPORTS_DIR / f"steam_{model_type}_recommendations.csv"
        cols = ["rank", "appid", "game_name", "rating", "probability", "genre_preference_score"]
        recommendations[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {out_path}")

    return recommendations, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="분류기 기반 개인 맞춤 게임 추천")
    parser.add_argument("--steam-id", required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) or ["lightgbm"], default="lightgbm")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    recs, _ = run_for_user(args.steam_id, model_type=args.model, top_n=args.top_n)
    print(recs[["rank", "game_name", "probability", "rating"]].to_string(index=False))
