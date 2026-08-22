"""
evaluation.py — 추천 품질 평가 (NDCG@K / Recall@K)

평가 방식: leave-out hold-out
  1. 각 유저의 플레이 게임 중 일부(기본 30%)를 숨긴다(test = 정답).
  2. 나머지(train)로만 모델을 만든다 (누수 방지).
  3. 모델이 안 가진 게임 전체를 점수순 정렬 → 숨긴 정답을 얼마나 위에 올리나.

NDCG (Normalized Discounted Cumulative Gain)
  - CG  : relevance의 단순 합 (순서 무시)
  - DCG : 순위 할인 적용  Σ rel_i / log2(i+1)
  - IDCG: 이상적 정렬(정답을 relevance 높은 순으로)의 DCG = 만점
  - NDCG: DCG / IDCG  (0~1, 유저별 크기 차이를 정규화해 비교 가능)
  relevance = 로그 플레이타임 (많이 한 게임일수록 높음)
"""

import numpy as np
import pandas as pd


def train_test_split(interactions, test_ratio=0.3, min_games=8, seed=42):
    """유저별로 플레이 게임의 일부를 test로 숨긴다.

    Returns
    -------
    train : DataFrame  (모델 학습용 상호작용)
    test  : dict {steamid: {appid: relevance(로그 플레이타임)}}
    """
    rng = np.random.default_rng(seed)
    train_parts, test = [], {}
    for sid, g in interactions.groupby("steamid"):
        if len(g) < min_games:
            continue
        n_test = max(1, int(len(g) * test_ratio))
        idx = rng.permutation(len(g))
        te = g.iloc[idx[:n_test]]
        tr = g.iloc[idx[n_test:]]
        test[sid] = dict(zip(te["appid"], np.log1p(te["playtime_hours"])))
        train_parts.append(tr)
    train = pd.concat(train_parts, ignore_index=True)
    return train, test


def _topk_index(scores, owned, k):
    """보유 게임 제외 후 상위 k개 appid."""
    s = scores.drop(index=[a for a in owned if a in scores.index], errors="ignore")
    return s.nlargest(k).index


def ndcg_at_k(scores, relevance, owned, k=10):
    top = _topk_index(scores, owned, k)
    dcg = sum(relevance.get(a, 0.0) / np.log2(i + 2) for i, a in enumerate(top))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(scores, relevance, owned, k=10):
    top = set(_topk_index(scores, owned, k))
    return len(top & set(relevance)) / len(relevance) if relevance else 0.0


def evaluate(models, ctx, test, k=10):
    """여러 모델을 같은 test로 채점해 비교표(DataFrame) 반환.

    models : dict {이름: recommend 함수}  (함수는 (user, ctx) -> pd.Series)
    """
    acc = {name: {"ndcg": [], "recall": []} for name in models}
    for sid, rel in test.items():
        owned = ctx["owned"].get(sid, set())
        for name, fn in models.items():
            scores = fn(sid, ctx)
            acc[name]["ndcg"].append(ndcg_at_k(scores, rel, owned, k))
            acc[name]["recall"].append(recall_at_k(scores, rel, owned, k))

    table = pd.DataFrame(
        {
            name: {
                f"NDCG@{k}": float(np.mean(v["ndcg"])),
                f"Recall@{k}": float(np.mean(v["recall"])),
            }
            for name, v in acc.items()
        }
    ).T
    return table.sort_values(f"NDCG@{k}", ascending=False)
