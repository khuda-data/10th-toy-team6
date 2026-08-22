"""
③ 유저 기반 협업 필터링 (User-CF)

원리: "나와 비슷하게 플레이한 사람들"이 즐긴 게임을 추천.
  1) 유저×게임 행렬(값=로그 플레이타임)에서 유저-유저 코사인 유사도
  2) 가장 비슷한 이웃 K명 선택
  3) 게임점수 = Σ(이웃 유사도 × 그 이웃의 게임 플레이타임)
사용 파일: cleaned_played_data.csv (ctx["matrix"], ctx["user_sim"])
"""

import pandas as pd

K_NEIGHBORS = 20


def recommend(user, ctx):
    mat = ctx["matrix"]
    if user not in mat.index:
        return pd.Series(0.0, index=ctx["all_games"])

    sims = ctx["user_sim"][user].drop(user)
    neighbors = sims.nlargest(K_NEIGHBORS)
    # 이웃들의 게임 플레이를 유사도로 가중합
    scores = mat.loc[neighbors.index].T.dot(neighbors)
    return scores.reindex(ctx["all_games"]).fillna(0.0)
