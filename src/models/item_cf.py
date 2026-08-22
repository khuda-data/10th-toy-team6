"""
④ 아이템 기반 협업 필터링 (Item-CF)

원리: "내가 한 게임과 같이 플레이되는 게임"을 추천. (User-CF와 축이 반대)
  1) 유저×게임 행렬을 전치해 게임-게임 코사인 유사도
  2) 게임점수 = 유저의 플레이 벡터 · 게임유사도 행렬
        = Σ(내가 한 게임들 × 그 게임과의 유사도)
사용 파일: cleaned_played_data.csv (ctx["matrix"], ctx["item_sim"])
"""

import pandas as pd


def recommend(user, ctx):
    mat = ctx["matrix"]
    if user not in mat.index:
        return pd.Series(0.0, index=ctx["all_games"])

    urow = mat.loc[user]                 # 유저의 게임별 플레이 벡터
    scores = ctx["item_sim"].dot(urow)   # 각 게임 = 내가 한 게임들과의 유사도 가중합
    return scores.reindex(ctx["all_games"]).fillna(0.0)
