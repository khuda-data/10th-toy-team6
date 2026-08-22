"""
⑥ Multi-Armed Bandit (MAB) 기반 추천

원리: 논문(이한우, 2022) 3장의 "분류-MAB" 방식을 그대로 재현.
  1) 유저의 태그 취향(ctx["taste"]) 중 rating이 가장 높은 태그 1개를 고른다.
     (논문 표3 실험에서 "태그 1개만 사용"이 4가지 방법 중 NDCG@10이 가장 높았음: 0.4617)
  2) 그 태그를 가진 게임들로 후보 목록을 만들고, 20개가 넘으면 인기도(게임 이용
     유저수) 상위 20개로 줄인다 — 논문은 "이용시간이 많은 순"으로 줄였지만, 여기서는
     추천 대상이 유저가 안 해본 게임이라 유저 본인 이용시간이 없으므로 인기도로 대체.
  3) 후보(=arm) 20개를 놓고 Epsilon-Greedy MAB를 시뮬레이션한다 — 논문 표4에서
     세 MAB(Epsilon-Greedy/UCB/Thompson Sampling) 중 가장 성능이 좋았던 방식(0.4653).
     reward는 논문처럼 "게임 이용 유저수"를 기반으로 한 확률(전체 유저 중 그 게임을
     플레이한 비율)의 베르누이 시행으로 시뮬레이션.
  4) N_PULLS번 당긴 뒤 추정 가치(Q) 순으로 점수를 매겨 반환.

사용 파일: 취향벡터(ctx["taste"]) + game_features_full(ctx["game_vectors"]) + 인기도(ctx["popularity"])

주의: MAB는 원래 실제 유저 반응을 reward로 받는 온라인 알고리즘이지만, 오프라인 평가에서는
      실제 반응이 없으므로 인기도를 reward 확률의 대리 신호로 사용해 시뮬레이션한다.
"""

import numpy as np
import pandas as pd

EPSILON = 0.1
N_PULLS = 200
CANDIDATE_SIZE = 20
SEED = 42


def _top_tag(user, ctx):
    """유저 취향(taste) 중 rating이 가장 높은 태그 1개(논문에서 가장 좋았던 설정)."""
    taste = ctx["taste"]
    if user not in taste.index:
        return None
    pref = taste.loc[user]
    tag_pref = pref[[c for c in pref.index if c.startswith("tag__")]]
    tag_pref = tag_pref[tag_pref > 0]
    if len(tag_pref) == 0:
        return None
    return tag_pref.idxmax()


def _candidate_games(user, ctx):
    """대표 태그 1개에 속한 게임 목록 → 20개로 축소(부족하면 인기도로 보충)."""
    gv = ctx["game_vectors"]
    pop = ctx["popularity"]
    tag = _top_tag(user, ctx)

    if tag is None:
        return pop.nlargest(CANDIDATE_SIZE).index.tolist()

    candidates = gv.index[gv[tag] > 0].tolist()

    if len(candidates) > CANDIDATE_SIZE:
        cand_pop = pop.reindex(candidates).fillna(0.0)
        candidates = cand_pop.nlargest(CANDIDATE_SIZE).index.tolist()
    elif len(candidates) < CANDIDATE_SIZE:
        extra = pop.drop(index=candidates, errors="ignore").nlargest(CANDIDATE_SIZE - len(candidates)).index.tolist()
        candidates = candidates + extra

    return candidates


def _epsilon_greedy(candidates, reward_p, seed):
    rng = np.random.default_rng(seed)
    counts = {a: 0 for a in candidates}
    values = {a: 0.0 for a in candidates}

    for _ in range(N_PULLS):
        if rng.random() < EPSILON or all(c == 0 for c in counts.values()):
            arm = candidates[rng.integers(len(candidates))]
        else:
            arm = max(values, key=values.get)
        reward = 1.0 if rng.random() < reward_p[arm] else 0.0
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

    return values


def recommend(user, ctx):
    candidates = _candidate_games(user, ctx)
    if not candidates:
        return pd.Series(0.0, index=ctx["all_games"])

    pop = ctx["popularity"]
    n_users = max(len(ctx["owned"]), 1)
    reward_p = {a: min(1.0, pop.get(a, 0) / n_users) for a in candidates}

    seed = (SEED + abs(hash(user))) % (2**32 - 1)
    values = _epsilon_greedy(candidates, reward_p, seed)

    scores = pd.Series(values)
    return scores.reindex(ctx["all_games"]).fillna(0.0)
