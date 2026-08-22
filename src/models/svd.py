"""
⑤ SVD 기반 행렬분해 (Matrix Factorization)

원리: 논문(이한우, 2022) 2장 1절의 편향 포함 SVD 공식을 그대로 사용.
    r_hat(u,i) = mu + b_u + b_i + p_u · q_i
  유저×게임 행렬(ctx["matrix"], 값=로그 플레이타임)에서 관측된 (u,i) 쌍만으로
  SGD로 mu/b_u/b_i/p_u/q_i를 학습해, 전체 유저×게임 예측 행렬을 만든다.
  User-CF/Item-CF가 "이웃"을 직접 못 찾는 희소 게임(1인 보유 등)에서도,
  latent factor를 통해 간접적으로 취향을 일반화할 수 있는 것이 SVD의 장점.

사용 파일: cleaned_played_data.csv (ctx["matrix"])

캐싱: recommend()가 유저마다 반복 호출되므로, 학습된 예측 행렬을 ctx에
      캐싱해(_svd_pred) 같은 ctx에 대해 한 번만 학습한다.
"""

import numpy as np
import pandas as pd

N_FACTORS = 30
N_EPOCHS = 20
LR = 0.005
REG = 0.05
SEED = 42


def _train(ctx):
    mat = ctx["matrix"]
    users = mat.index.to_numpy()
    items = mat.columns.to_numpy()
    R = mat.values

    rows, cols = np.nonzero(R)
    vals = R[rows, cols]

    rng = np.random.default_rng(SEED)
    mu = float(vals.mean())
    bu = np.zeros(len(users))
    bi = np.zeros(len(items))
    P = rng.normal(0, 0.1, (len(users), N_FACTORS))
    Q = rng.normal(0, 0.1, (len(items), N_FACTORS))

    order = np.arange(len(vals))
    for _ in range(N_EPOCHS):
        rng.shuffle(order)
        for idx in order:
            u, i, r = rows[idx], cols[idx], vals[idx]
            pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
            e = r - pred
            bu[u] += LR * (e - REG * bu[u])
            bi[i] += LR * (e - REG * bi[i])
            p_old = P[u].copy()
            P[u] += LR * (e * Q[i] - REG * P[u])
            Q[i] += LR * (e * p_old - REG * Q[i])

    pred_matrix = mu + bu[:, None] + bi[None, :] + P @ Q.T
    return pd.DataFrame(pred_matrix, index=users, columns=items)


def recommend(user, ctx):
    if "_svd_pred" not in ctx:
        ctx["_svd_pred"] = _train(ctx)

    pred = ctx["_svd_pred"]
    if user not in pred.index:
        return pd.Series(0.0, index=ctx["all_games"])
    return pred.loc[user].reindex(ctx["all_games"]).fillna(0.0)
