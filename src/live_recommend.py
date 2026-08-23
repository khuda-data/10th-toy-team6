"""
신규 유저(cold-start) 실시간 추천

`ctx["taste"]`에 없는 steamid(팀 공식 779명 외부)를 위한 것. Steam API로 보유 게임을
그 자리에서 가져와 `models/content.py`와 동일한 방식(취향벡터 코사인 유사도 + Value Score)
으로 취향벡터를 즉석 계산해 추천한다. 이미 등록된 유저는 `models.content.recommend()`를
그대로 쓰면 되고, 이 모듈은 그 함수가 처리 못 하는 케이스만 담당한다.

Content 방식이 이 용도에 맞는 이유: CF(User-CF/Item-CF)는 새 유저를 넣으려면 전체
유저-유저/게임-게임 유사도 행렬을 다시 계산해야 하지만, Content는 그 유저의 게임 목록 +
플레이타임만 있으면 기존 게임 벡터(`ctx["game_vectors"]`)와 대조해 바로 추천할 수 있다.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from steam_client import fetch_owned_games

# models/content.py와 동일한 가중치 (일관성 유지)
ALPHA = 0.6  # 취향 유사도
BETA = 0.2  # 평점
GAMMA = 0.2  # 인기도


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def build_live_taste(owned_games: pd.DataFrame, ctx):
    """실시간으로 가져온 보유 게임으로 취향벡터를 만든다.

    Returns
    -------
    taste : pd.Series (42차원, 장르+태그)
    n_matched, n_total : 카탈로그와 매칭된 게임 수 / 전체 보유 게임 수
        (카탈로그에 없는 게임은 장르·태그를 몰라 취향 계산에서 자연히 제외됨)
    """
    gv = ctx["game_vectors"]
    feat_cols = ctx["feat_cols"]

    merged = owned_games.merge(gv.reset_index(), on="appid", how="inner")
    n_matched, n_total = len(merged), len(owned_games)

    if n_matched == 0:
        return pd.Series(0.0, index=feat_cols), n_matched, n_total

    w = np.log1p(merged["playtime_hours"])
    taste_values = (merged[feat_cols].values * w.values[:, None]).sum(axis=0)
    return pd.Series(taste_values, index=feat_cols), n_matched, n_total


def recommend_live(steamid: str, ctx):
    """Steam API 실시간 조회 + Content 방식 추천.

    Returns
    -------
    owned_games : pd.DataFrame[appid, game_name, playtime_hours]
    scores : pd.Series (appid -> 추천 점수, 보유 게임 제외, 내림차순 아님)
    n_matched, n_total : 카탈로그 매칭 커버리지
    """
    owned_games = fetch_owned_games(steamid)
    owned_appids = set(owned_games["appid"])

    taste, n_matched, n_total = build_live_taste(owned_games, ctx)

    gv = ctx["game_vectors"]
    uvec = taste.values.reshape(1, -1)
    sim = pd.Series(cosine_similarity(uvec, gv.values)[0], index=gv.index)

    rating = ctx["game_numeric"]["rating_by_reviews"].reindex(gv.index).fillna(0.0) / 100.0
    pop = _minmax(ctx["popularity"].reindex(gv.index).fillna(0.0))

    score = ALPHA * _minmax(sim) + BETA * rating + GAMMA * pop
    score = score.drop(index=[a for a in owned_appids if a in score.index], errors="ignore")

    return owned_games, score, n_matched, n_total
