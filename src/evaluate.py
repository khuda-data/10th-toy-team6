# -*- coding: utf-8 -*-
"""
추천 모델 공통 평가 모듈.

Leave-One-Out 방식:
  유저별 상호작용 중 1개(가능하면 최근 플레이 게임)를 정답으로 숨기고,
  (정답 1개 + 안 해본 게임 중 무작위 N개)를 후보로 랭킹시켜 성능을 측정한다.

지표: HitRate@K, Precision@K, Recall@K, NDCG@K
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def leave_one_out_split(user_games, min_interactions=3):
    test_rows, train_parts = [], []
    for steamid, g in user_games.groupby("steamid"):
        if len(g) < min_interactions:
            train_parts.append(g)
            continue
        if "recent_playtime_hours" in g.columns and g["recent_playtime_hours"].max() > 0:
            test_i = g["recent_playtime_hours"].idxmax()
        else:
            test_i = g.sample(1, random_state=42).index[0]
        test_rows.append(g.loc[test_i])
        train_parts.append(g.drop(index=test_i))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)
    return train_df, test_df


def evaluate_model(model, test_df, train_df, all_items, k=10, n_neg=99):
    played_by_user = train_df.groupby("steamid")["appid"].apply(set).to_dict()
    all_items = np.array(all_items)

    hits, precisions, recalls, ndcgs = [], [], [], []

    for _, row in test_df.iterrows():
        steamid = row["steamid"]
        true_item = row["appid"]
        played = played_by_user.get(steamid, set())
        candidate_pool = np.setdiff1d(all_items, np.array(list(played) + [true_item]), assume_unique=False)
        if len(candidate_pool) == 0:
            continue
        negs = RNG.choice(candidate_pool, size=min(n_neg, len(candidate_pool)), replace=False)
        candidates = np.concatenate([[true_item], negs])

        scores = model.score(steamid, candidates)
        ranked = candidates[np.argsort(-scores)]
        topk = ranked[:k]

        hit = int(true_item in topk)
        hits.append(hit)
        precisions.append(hit / k)
        recalls.append(hit)
        if hit:
            rank = np.where(topk == true_item)[0][0] + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)

    return {
        f"HitRate@{k}": np.mean(hits) if hits else 0.0,
        f"Precision@{k}": np.mean(precisions) if precisions else 0.0,
        f"Recall@{k}": np.mean(recalls) if recalls else 0.0,
        f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0,
        "n_eval_users": len(hits),
    }
