"""
① 인기도 (Popularity) — baseline

정의: 게임을 플레이한 고유 유저 수가 많을수록 높은 점수. 개인화 없음(모두 동일).
사용 파일: cleaned_played_data.csv (ctx["popularity"]로 준비됨)
역할: 다른 모델이 이걸 넘어야 "진짜 실력". 반드시 비교군으로 둔다.
"""

import pandas as pd


def recommend(user, ctx):
    """유저에게 게임별 점수(인기도)를 반환. 모든 유저에게 동일."""
    pop = ctx["popularity"].reindex(ctx["all_games"]).fillna(0.0)
    return pop.astype(float)
