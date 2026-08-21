# -*- coding: utf-8 -*-
"""
협업 필터링(Collaborative Filtering) 계열 추천 모델.

  - UserCFRecommender: 유저-유저 kNN (취향이 비슷한 유저가 한 게임을 추천)
  - ItemCFRecommender: 아이템-아이템 kNN (같이 플레이되는 게임을 추천)
  - ALSRecommender: implicit 라이브러리의 Alternating Least Squares (행렬 분해)
  - HybridRecommender: ContentBased + ALS 가중합
"""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def build_interaction_matrix(user_games):
    """유저 x 아이템 sparse 행렬 (값 = log1p(playtime_hours))을 만든다.

    Parameters
    ----------
    user_games : pd.DataFrame
        steamid, appid, playtime_hours 컬럼을 포함한 롱포맷 상호작용 데이터.

    Returns
    -------
    mat, users, items, user_idx, item_idx
    """
    users = user_games["steamid"].unique()
    items = user_games["appid"].unique()
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {g: i for i, g in enumerate(items)}

    rows = user_games["steamid"].map(user_idx).to_numpy()
    cols = user_games["appid"].map(item_idx).to_numpy()
    vals = np.log1p(user_games["playtime_hours"].to_numpy())

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items, user_idx, item_idx


class UserCFRecommender:
    name = "UserCF"

    def fit(self, mat, user_idx, item_idx, n_neighbors=30):
        self.mat = mat
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.model = NearestNeighbors(metric="cosine", n_neighbors=min(n_neighbors, mat.shape[0]))
        self.model.fit(mat)

    def score(self, steamid, candidate_items):
        if steamid not in self.user_idx:
            return np.zeros(len(candidate_items))
        uidx = self.user_idx[steamid]
        dist, neigh = self.model.kneighbors(self.mat[uidx], n_neighbors=min(31, self.mat.shape[0]))
        dist, neigh = dist.flatten(), neigh.flatten()
        sims = 1 - dist
        neigh_mat = self.mat[neigh]
        weighted = np.asarray(neigh_mat.T.dot(sims)).flatten()
        total_sim = max(sims.sum(), 1e-9)
        item_scores = weighted / total_sim
        return np.array([item_scores[self.item_idx[i]] if i in self.item_idx else 0.0 for i in candidate_items])


class ItemCFRecommender:
    name = "ItemCF"

    def fit(self, mat, user_idx, item_idx, n_neighbors=30):
        self.mat_t = mat.T.tocsr()
        self.mat = mat
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.model = NearestNeighbors(metric="cosine", n_neighbors=min(n_neighbors, self.mat_t.shape[0]))
        self.model.fit(self.mat_t)

    def score(self, steamid, candidate_items):
        if steamid not in self.user_idx:
            return np.zeros(len(candidate_items))
        uidx = self.user_idx[steamid]
        interacted = self.mat[uidx].indices
        if len(interacted) == 0:
            return np.zeros(len(candidate_items))
        n_neigh = min(21, self.mat_t.shape[0])
        dist, neigh = self.model.kneighbors(self.mat_t[interacted], n_neighbors=n_neigh)
        sims = 1 - dist
        scores = np.zeros(self.mat_t.shape[0])
        for row_sims, row_neigh in zip(sims, neigh):
            scores[row_neigh] += row_sims
        return np.array([scores[self.item_idx[i]] if i in self.item_idx else 0.0 for i in candidate_items])


class ALSRecommender:
    """implicit.als.AlternatingLeastSquares 기반 행렬 분해."""

    name = "ALS (MF)"

    def fit(self, mat, user_idx, item_idx, factors=32, regularization=0.05, iterations=20):
        import implicit

        self.user_idx = user_idx
        self.item_idx = item_idx
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations, random_state=42,
        )
        self.conf_mat = (mat * 3).tocsr()
        self.model.fit(self.conf_mat)

    def score(self, steamid, candidate_items):
        if steamid not in self.user_idx:
            return np.zeros(len(candidate_items))
        ufactor = self.model.user_factors[self.user_idx[steamid]]
        out = []
        for item in candidate_items:
            j = self.item_idx.get(item)
            out.append(float(ufactor @ self.model.item_factors[j]) if j is not None else 0.0)
        return np.array(out)


class HybridRecommender:
    """콘텐츠 기반 점수와 ALS 점수를 정규화 후 가중합."""

    name = "Hybrid (Content+ALS)"

    def __init__(self, content_model, als_model, alpha=0.5):
        self.content_model = content_model
        self.als_model = als_model
        self.alpha = alpha

    def score(self, steamid, candidate_items):
        c = self.content_model.score(steamid, candidate_items)
        a = self.als_model.score(steamid, candidate_items)

        def _norm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

        return self.alpha * _norm(c) + (1 - self.alpha) * _norm(a)
