"""
data.py — 데이터 로드 + 전처리 (모든 모델 공통 재료 준비)

CSV 파일을 새로 만들지 않는다. 이미 있는 CSV를 읽어서, 모델이 쓸 수 있는
형태(행렬·벡터·유사도 등)로 메모리에서 가공해 하나의 ctx(dict)로 반환한다.
모든 추천 모델(popularity/content/user_cf/item_cf)은 이 ctx만 받아서 쓴다.

평가 시 데이터 누수를 막기 위해, ctx는 '주어진 상호작용(train)'만으로
행렬·취향벡터·유사도를 다시 계산한다. 전체 데이터로 부르면 실서비스용 ctx가,
train만 넣으면 평가용 ctx가 된다.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 기본 데이터 경로 (repo 루트/data/processed)
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(data_dir=DATA_DIR):
    """원본 CSV들을 그대로 읽어서 반환."""
    data_dir = Path(data_dir)
    interactions = pd.read_csv(data_dir / "cleaned_played_data.csv")
    game = pd.read_csv(data_dir / "game_features_full.csv")
    ug = pd.read_csv(data_dir / "user_genre_preference_normalized_clean.csv")
    ut = pd.read_csv(data_dir / "user_tag_preference_top30_normalized_clean.csv")
    return interactions, game, ug, ut


def feature_columns(ug, ut):
    """콘텐츠 기반이 쓰는 42차원 컬럼(장르12 + 태그30)."""
    genre_cols = [c for c in ug.columns if c.startswith("genre__")]
    tag_cols = [c for c in ut.columns if c.startswith("tag__")]
    return genre_cols + tag_cols


def _build_taste(interactions, game_vectors, feat_cols):
    """유저별 취향벡터를 상호작용에서 다시 계산 (누수 방지).

    유저벡터 = Σ(게임 feature벡터 × 로그 플레이타임), 그 유저가 '한' 게임만.
    (전체 데이터로 계산하면 팀원의 _clean 취향벡터와 같은 의미)
    """
    gv = game_vectors.reset_index()  # appid + 42컬럼
    m = interactions.merge(gv, on="appid", how="inner")
    for c in feat_cols:
        m[c] = m[c] * m["w"]
    taste = m.groupby("steamid")[feat_cols].sum()
    return taste


def build_context(interactions, game, ug, ut, precompute_sim=True):
    """상호작용(subset 가능) + 게임/취향 원본으로 공통 재료(ctx)를 만든다."""
    interactions = interactions.copy()
    # 로그 변환 (없으면 생성)
    if "w" not in interactions.columns:
        interactions["w"] = np.log1p(interactions["playtime_hours"])

    feat_cols = feature_columns(ug, ut)

    # 게임 벡터 (42차원, 컬럼 정렬)
    game_vectors = game.set_index("appid").reindex(columns=feat_cols).fillna(0.0)

    # 게임 수치형 (Value Score용): 평점
    game_numeric = game.set_index("appid")[["rating_by_reviews"]].copy()

    # 유저 취향벡터 (train 상호작용으로 재계산)
    taste = _build_taste(interactions, game_vectors, feat_cols)

    # 유저×게임 행렬 (CF용)
    matrix = interactions.pivot_table(
        index="steamid", columns="appid", values="w", fill_value=0.0
    )

    # 인기도 = 게임을 플레이한 고유 유저 수
    popularity = interactions.groupby("appid")["steamid"].nunique()

    # 유저별 보유(플레이) 게임 집합 — 추천에서 제외
    owned = interactions.groupby("steamid")["appid"].apply(set).to_dict()

    # appid → 게임이름
    names = interactions.drop_duplicates("appid").set_index("appid")["game_name"]

    ctx = {
        "interactions": interactions,
        "matrix": matrix,
        "taste": taste,
        "game_vectors": game_vectors,
        "game_numeric": game_numeric,
        "popularity": popularity,
        "owned": owned,
        "names": names,
        "feat_cols": feat_cols,
        "all_games": list(game_vectors.index),
    }

    # CF 유사도 행렬 미리 계산 (train 기준이라 누수 없음)
    if precompute_sim:
        ctx["user_sim"] = pd.DataFrame(
            cosine_similarity(matrix), index=matrix.index, columns=matrix.index
        )
        ctx["item_sim"] = pd.DataFrame(
            cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns
        )

    return ctx


def load_context(data_dir=DATA_DIR, official_users_only=True):
    """전체 데이터로 실서비스용 ctx를 만든다 (편의 함수)."""
    interactions, game, ug, ut = load_raw(data_dir)
    if official_users_only:
        interactions = interactions[interactions["steamid"].isin(set(ug["steamid"]))]
    return build_context(interactions, game, ug, ut)
