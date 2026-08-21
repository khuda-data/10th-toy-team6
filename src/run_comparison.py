# -*- coding: utf-8 -*-
"""
전체 추천 모델(Popularity / ContentBased / UserCF / ItemCF / ALS / Hybrid)을
동일한 데이터와 평가 방식(Leave-One-Out)으로 비교하는 진입점 스크립트.

실행:
    cd src && python run_comparison.py

산출물:
    reports/recommender_comparison.csv
    reports/recommender_comparison.png
"""

from pathlib import Path

import pandas as pd

from recommenders.content_based import PopularityRecommender, ContentBasedRecommender
from recommenders.collaborative import (
    build_interaction_matrix, UserCFRecommender, ItemCFRecommender,
    ALSRecommender, HybridRecommender,
)
from evaluate import leave_one_out_split, evaluate_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    user_games = pd.read_csv(DATA_DIR / "steam_user_games_classified.csv", encoding="utf-8-sig")
    game_features = pd.read_csv(DATA_DIR / "game_features.csv", encoding="utf-8-sig")
    user_features = pd.read_csv(DATA_DIR / "user_features.csv", encoding="utf-8-sig")

    user_games["playtime_hours"] = pd.to_numeric(user_games["playtime_hours"], errors="coerce").fillna(0)
    user_games = user_games[user_games["playtime_hours"] > 0].copy()
    user_games["appid"] = user_games["appid"].astype(str)
    game_features["appid"] = game_features["appid"].astype(str)
    return user_games, game_features, user_features


def main():
    print("[1/5] 데이터 로드...")
    user_games, game_features, user_features = load_data()

    print("[2/5] Leave-One-Out split...")
    train_df, test_df = leave_one_out_split(user_games)
    print(f"  train interactions: {len(train_df):,} / test users: {len(test_df):,}")

    print("[3/5] 상호작용 행렬 구성...")
    mat, users, items, user_idx, item_idx = build_interaction_matrix(train_df)
    all_items = list(item_idx.keys())

    print("[4/5] 모델 학습...")
    pop = PopularityRecommender()
    pop.fit(train_df, item_idx)

    content = ContentBasedRecommender()
    content.fit(user_features, game_features)

    user_cf = UserCFRecommender()
    user_cf.fit(mat, user_idx, item_idx)

    item_cf = ItemCFRecommender()
    item_cf.fit(mat, user_idx, item_idx)

    als = ALSRecommender()
    als.fit(mat, user_idx, item_idx)

    hybrid = HybridRecommender(content, als, alpha=0.4)

    print("[5/5] 평가...")
    results = []
    for m in [pop, content, user_cf, item_cf, als, hybrid]:
        res = evaluate_model(m, test_df, train_df, all_items)
        res["model"] = m.name
        results.append(res)
        print(f"  {m.name:25s} -> {res}")

    result_df = pd.DataFrame(results).set_index("model")
    result_df.to_csv(REPORT_DIR / "recommender_comparison.csv", encoding="utf-8-sig")
    print("\n=== 최종 비교 ===")
    print(result_df.round(4))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    result_df[["HitRate@10", "NDCG@10"]].plot(kind="bar", ax=ax)
    ax.set_title("Recommender Comparison (Leave-One-Out, K=10)")
    ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "recommender_comparison.png", dpi=150)
    print(f"\n[저장 완료] {REPORT_DIR / 'recommender_comparison.csv'}")
    print(f"[저장 완료] {REPORT_DIR / 'recommender_comparison.png'}")


if __name__ == "__main__":
    main()
