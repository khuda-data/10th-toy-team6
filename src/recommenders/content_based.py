# -*- coding: utf-8 -*-
"""
비개인화 베이스라인(Popularity)과 콘텐츠 기반 추천(ContentBased).

ContentBasedRecommender는 최종모델.md ③~④에서 설명한 방식을 그대로 구현한다:
  유저 장르 선호 벡터 <-> 게임 장르 벡터의 코사인 유사도.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    """전체 유저가 많이 플레이한 게임 순으로 추천하는 비개인화 베이스라인."""

    name = "Popularity"

    def fit(self, train_df, item_idx):
        pop = train_df.groupby("appid")["steamid"].nunique()
        self.pop = pop.reindex(item_idx.keys()).fillna(0)

    def score(self, steamid, candidate_items):
        return np.array([self.pop.get(i, 0) for i in candidate_items], dtype=float)


class ContentBasedRecommender:
    """유저 장르 선호 벡터 vs 게임 장르 벡터 코사인 유사도 기반 추천."""

    name = "ContentBased"

    def fit(self, user_features_df, game_features_df):
        self.user_features_df = user_features_df.set_index("steamid")
        self.game_features_df = game_features_df.set_index("appid")
        genre_cols = [c for c in game_features_df.columns if c.startswith("genre_")]
        user_genre_cols = [f"user_{c}" for c in genre_cols]
        self.genre_cols = genre_cols
        self.user_genre_cols = [c for c in user_genre_cols if c in self.user_features_df.columns]

    def score(self, steamid, candidate_items):
        if steamid not in self.user_features_df.index or not self.user_genre_cols:
            return np.zeros(len(candidate_items))
        uvec = self.user_features_df.loc[steamid, self.user_genre_cols].to_numpy(dtype=float).reshape(1, -1)
        gcols = [c.replace("user_", "") for c in self.user_genre_cols]
        rows = []
        for item in candidate_items:
            if item in self.game_features_df.index:
                rows.append(self.game_features_df.loc[item, gcols].to_numpy(dtype=float))
            else:
                rows.append(np.zeros(len(gcols)))
        gmat = np.vstack(rows)
        return cosine_similarity(uvec, gmat).flatten()
