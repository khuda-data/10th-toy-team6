# -*- coding: utf-8 -*-
"""
전처리 끝난 데이터를 기반으로 게임/유저 feature를 확장 생성하는 모듈.

기존에는 genre one-hot(12개) + rating 정도만 사용했는데,
여기서는 다음을 추가한다.

[게임 feature]
- 장르 multi-hot (12개, 기존 로직 재사용)
- 태그 기반 feature: 상위 N개 태그에 대한 multi-hot (장르보다 세분화된 취향 반영)
- 수치형 feature: price, review_count, positive_ratio, metacritic, release_year, game_age_years
- 인기도 feature: log1p(review_count), log1p(recommendations)
- 카테고리 feature: 싱글/멀티/온라인 co-op 여부 (categories 텍스트 파싱)

[유저 feature]
- 장르 선호 벡터 (already existed as user_genre_weights.csv, 재계산 로직 포함)
- 태그 선호 벡터 (playtime 가중 평균)
- 유저 통계: 보유 게임 수, 총 playtime, 평균 playtime, 장르 다양성(엔트로피)
- 최근성 반영: recent_playtime_hours 비중

실행:
    python features.py
산출물:
    data/processed/game_features.csv
    data/processed/user_features.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TOP500_CSV = DATA_DIR / "steam_top500_games_classified.csv"
USER_GAMES_CSV = DATA_DIR / "steam_user_games_classified.csv"

GENRE_KEYWORDS = {
    "Action": ["action", "hack and slash", "beat 'em up", "beat em up", "fighting", "martial arts", "character action"],
    "Adventure": ["adventure", "point & click", "point and click", "exploration", "walking simulator", "story rich", "interactive fiction"],
    "RPG": ["rpg", "action rpg", "action-rpg", "jrpg", "crpg", "souls-like", "soulslike", "party-based rpg", "turn-based rpg", "turn based rpg", "old school rpg", "roguelike rpg", "role-playing"],
    "Strategy": ["strategy", "real-time strategy", "real time strategy", "rts", "turn-based strategy", "turn based strategy", "4x", "grand strategy", "tactics", "tactical", "tower defense", "tower defence"],
    "Simulation": ["simulation", "sim", "life sim", "farming sim", "city builder", "management", "building", "automation", "sandbox"],
    "Shooter": ["shooter", "fps", "first-person shooter", "first person shooter", "third-person shooter", "third person shooter", "tps", "tactical shooter", "hero shooter", "looter shooter", "bullet hell"],
    "Sports": ["sports", "football", "soccer", "basketball", "baseball", "tennis", "golf", "hockey", "volleyball", "wrestling"],
    "Racing": ["racing", "racer", "driving", "automobile sim", "car"],
    "Horror": ["horror", "survival horror", "psychological horror"],
    "Survival": ["survival", "survival crafting", "open world survival craft", "crafting", "base building"],
    "Platformer": ["platformer", "2d platformer", "3d platformer", "metroidvania"],
    "Puzzle": ["puzzle", "logic", "match 3", "match-3"],
    "Casual": ["casual", "relaxing", "family friendly"],
}
GENRES = list(GENRE_KEYWORDS.keys())

TOP_TAG_N = 40  # 태그 feature로 사용할 상위 태그 개수


def normalize_text(v):
    if pd.isna(v):
        return ""
    v = str(v).lower().strip()
    return re.sub(r"\s+", " ", v)


def split_values(v):
    if v is None or pd.isna(v):
        return []
    v = str(v).strip()
    if not v or v.lower() == "unknown":
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def detect_genres(values):
    detected = []
    for value in values:
        value = normalize_text(value)
        if not value:
            continue
        for genre, keywords in GENRE_KEYWORDS.items():
            for kw in keywords:
                kw = normalize_text(kw)
                if value == kw or kw in value:
                    if genre not in detected:
                        detected.append(genre)
                    break
    return detected


def determine_genres(genre, tags):
    g = detect_genres(split_values(genre))
    if g:
        return g
    g = detect_genres(split_values(tags))
    if g:
        return g
    return ["Unknown"]


def parse_price(series):
    """price 컬럼은 top500은 정수(원 단위로 보임), user_games는 'price(원)' 문자열.
    둘 다 숫자로 강제 변환, 실패시 NaN."""
    return pd.to_numeric(series, errors="coerce")


def parse_release_year(series):
    # "7 Jul, 2026" 같은 포맷에서 연도만 추출
    return series.astype(str).str.extract(r"(\d{4})").astype(float)


def build_top_tags(df, tag_col="tags", n=TOP_TAG_N):
    from collections import Counter
    counter = Counter()
    for tags in df[tag_col]:
        for t in split_values(tags):
            counter[normalize_text(t)] += 1
    return [t for t, _ in counter.most_common(n)]


def build_game_features(df, top_tags, id_col="appid", name_col="game_name"):
    rows = []
    for _, row in df.iterrows():
        genres = determine_genres(row.get("genre", ""), row.get("tags", ""))
        tag_values = set(normalize_text(t) for t in split_values(row.get("tags", "")))
        cats = normalize_text(row.get("categories", row.get("steam_categories", "")))

        rec = {
            "appid": str(row[id_col]),
            "game_name": row[name_col],
        }
        for g in GENRES:
            rec[f"genre_{g}"] = int(g in genres)
        for t in top_tags:
            rec[f"tag_{t}"] = int(t in tag_values)

        rec["is_singleplayer"] = int("single-player" in cats)
        rec["is_multiplayer"] = int("multi-player" in cats or "online co-op" in cats)
        rec["is_coop"] = int("co-op" in cats)

        price = parse_price(pd.Series([row.get("price", row.get("price(원)", np.nan))])).iloc[0]
        rec["price"] = 0.0 if pd.isna(price) else float(price)

        rating = pd.to_numeric(row.get("rating", row.get("rating_by_reviews", np.nan)), errors="coerce")
        rec["rating"] = 0.0 if pd.isna(rating) else float(rating)

        review_count = pd.to_numeric(row.get("review_count", row.get("total_reviews", np.nan)), errors="coerce")
        rec["review_count"] = 0.0 if pd.isna(review_count) else float(review_count)
        rec["log_review_count"] = np.log1p(rec["review_count"])

        pos = pd.to_numeric(row.get("positive_reviews", np.nan), errors="coerce")
        neg = pd.to_numeric(row.get("negative_reviews", np.nan), errors="coerce")
        if pd.notna(pos) and pd.notna(neg) and (pos + neg) > 0:
            rec["positive_ratio"] = float(pos / (pos + neg))
        else:
            rec["positive_ratio"] = rec["rating"] / 100.0 if rec["rating"] > 1 else rec["rating"]

        year = parse_release_year(pd.Series([row.get("release_date", np.nan)])).iloc[0, 0]
        rec["release_year"] = float(year) if pd.notna(year) else np.nan
        rec["game_age_years"] = (2026 - rec["release_year"]) if pd.notna(rec["release_year"]) else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    # 결측치는 중앙값으로 보정 (release_year, game_age_years)
    for col in ["release_year", "game_age_years"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def build_user_features(user_games_df, game_features_df, top_tags):
    """steam_user_games_classified.csv (long format) -> 유저별 feature"""
    gdf = user_games_df.copy()
    gdf["playtime_hours"] = pd.to_numeric(gdf["playtime_hours"], errors="coerce").fillna(0)
    gdf = gdf[gdf["playtime_hours"] > 0]

    rows = []
    for steamid, g in gdf.groupby("steamid"):
        log_pt = np.log1p(g["playtime_hours"].to_numpy())
        total = log_pt.sum()
        weights = log_pt / total if total > 0 else np.ones(len(g)) / len(g)

        genre_pref = {gname: 0.0 for gname in GENRES}
        tag_pref = {t: 0.0 for t in top_tags}

        for (idx, row), w in zip(g.iterrows(), weights):
            genres = determine_genres(row.get("genre", ""), row.get("tags", ""))
            for gname in genres:
                if gname in genre_pref:
                    genre_pref[gname] += w
            tag_values = set(normalize_text(t) for t in split_values(row.get("tags", "")))
            for t in top_tags:
                if t in tag_values:
                    tag_pref[t] += w

        gp_total = sum(genre_pref.values())
        if gp_total > 0:
            genre_pref = {k: v / gp_total for k, v in genre_pref.items()}
        tp_total = sum(tag_pref.values())
        if tp_total > 0:
            tag_pref = {k: v / tp_total for k, v in tag_pref.items()}

        # 장르 다양성 (엔트로피, 값이 클수록 다양한 장르를 즐김)
        probs = np.array(list(genre_pref.values()))
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log(probs)).sum()) if len(probs) > 0 else 0.0

        recent_ratio = (
            pd.to_numeric(g.get("recent_playtime_hours", 0), errors="coerce").fillna(0).sum()
            / max(g["playtime_hours"].sum(), 1e-9)
        )

        rec = {
            "steamid": steamid,
            "num_games": len(g),
            "total_playtime_hours": float(g["playtime_hours"].sum()),
            "avg_playtime_hours": float(g["playtime_hours"].mean()),
            "genre_entropy": entropy,
            "recent_playtime_ratio": float(recent_ratio),
        }
        rec.update({f"user_genre_{k}": v for k, v in genre_pref.items()})
        rec.update({f"user_tag_{k}": v for k, v in tag_pref.items()})
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    top500 = pd.read_csv(TOP500_CSV, encoding="cp949")
    user_games = pd.read_csv(USER_GAMES_CSV, encoding="utf-8-sig")

    top_tags = build_top_tags(pd.concat([
        top500.rename(columns={"tags": "tags"})[["tags"]],
        user_games[["tags"]],
    ], ignore_index=True))

    print(f"[features] 상위 태그 {len(top_tags)}개 선정")

    game_features = build_game_features(top500, top_tags)
    game_features.to_csv(DATA_DIR / "game_features.csv", index=False, encoding="utf-8-sig")
    print(f"[features] game_features.csv 저장 ({game_features.shape})")

    user_features = build_user_features(user_games, game_features, top_tags)
    user_features.to_csv(DATA_DIR / "user_features.csv", index=False, encoding="utf-8-sig")
    print(f"[features] user_features.csv 저장 ({user_features.shape})")


if __name__ == "__main__":
    main()
