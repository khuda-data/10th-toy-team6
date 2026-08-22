"""
② 콘텐츠 기반 (Content-based) + Value Score

원리
  1) 취향 매칭: 유저 42차원 취향벡터 ↔ 게임 42차원 장르·태그 벡터의 코사인 유사도
  2) Value Score: 유사도만 쓰면 같은 장르 게임을 구별 못 함(동점) →
     평점·인기도(수치형 feature)를 항으로 더해 동점을 깬다.
        점수 = ALPHA*유사도 + BETA*평점 + GAMMA*인기도   (각 항 0~1 정규화)

수치형 feature 방침
  - 평점(rating_by_reviews), 인기도: 사용 (Value Score 항)
  - 가격: 제외 (품질 신호 아님) — TODO: 나중에 δ항으로 추가해 NDCG 개선 테스트
  - 메타크리틱: 제외 (결측 많음)

사용 파일: 취향벡터 2개(merge=ctx["taste"]) + game_features_full(ctx["game_vectors"], ctx["game_numeric"]) + 인기도
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Value Score 가중치 (합=1). NDCG로 튜닝 대상.
ALPHA = 0.6   # 취향 유사도
BETA = 0.2    # 평점
GAMMA = 0.2   # 인기도


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def recommend(user, ctx):
    gv = ctx["game_vectors"]
    taste = ctx["taste"]
    if user not in taste.index:
        return pd.Series(0.0, index=ctx["all_games"])

    # 1) 취향 유사도
    uvec = taste.loc[user].values.reshape(1, -1)
    sim = pd.Series(cosine_similarity(uvec, gv.values)[0], index=gv.index)

    # 2) Value Score 항 (0~1)
    rating = (ctx["game_numeric"]["rating_by_reviews"].reindex(gv.index).fillna(0.0) / 100.0)
    pop = _minmax(ctx["popularity"].reindex(gv.index).fillna(0.0))

    score = ALPHA * _minmax(sim) + BETA * rating + GAMMA * pop
    return score.reindex(ctx["all_games"]).fillna(0.0)
