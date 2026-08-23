"""
Content 기반 게임 추천 데모 (Streamlit)

탭1: 팀 공식 유저(779명) 중 steamid 선택 → 미리 계산된 취향벡터로 추천.
탭2: 팀 데이터에 없는 새 steamid → Steam API로 실시간 조회해 그 자리에서 추천
     (STEAM_API_KEY 환경변수 필요, https://steamcommunity.com/dev/apikey 에서 발급).

실행:
    streamlit run app/recommend_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 로컬은 .env, Streamlit Cloud 배포 시엔 앱 설정의 Secrets(STEAM_API_KEY)를 그대로 사용.
# secrets.toml이 없는 로컬 실행에서는 st.secrets 접근 시 예외가 날 수 있어 무시한다.
if "STEAM_API_KEY" not in os.environ:
    try:
        os.environ["STEAM_API_KEY"] = st.secrets["STEAM_API_KEY"]
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data
import live_recommend
import steam_client
from models import content

st.set_page_config(page_title="게임 추천 데모 (Content 기반)", page_icon="🎮")


@st.cache_resource
def load_ctx():
    return data.load_context()


ctx = load_ctx()
users = sorted(ctx["owned"].keys())

st.title("🎮 Content 기반 게임 추천 데모")
st.caption(
    f"팀 공식 유저 {len(users)}명 · 게임 {len(ctx['all_games'])}개 "
    f"(`docs/ndcg_diversity_tradeoff.md` 최종 선택 모델)"
)

tab_known, tab_live = st.tabs(["등록된 유저", "새 유저 (Steam API 실시간 조회)"])

with tab_known:
    steamid_input = st.selectbox("유저 steamid 선택", options=users, index=0)

    if steamid_input:
        uid = int(steamid_input)
        owned = ctx["owned"].get(uid, set())

        played = ctx["interactions"]
        top5_played = (
            played[played["steamid"] == uid]
            .nlargest(5, "playtime_hours")[["game_name", "playtime_hours"]]
        )

        scores = content.recommend(uid, ctx)
        scores = scores.drop(index=[a for a in owned if a in scores.index], errors="ignore")
        top5_rec = scores.nlargest(5)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("많이 플레이한 게임")
            for row in top5_played.itertuples():
                st.write(f"- {row.game_name} ({row.playtime_hours:.0f}h)")

        with col2:
            st.subheader("추천 Top-5 (Content 기반)")
            for appid, score in top5_rec.items():
                name = ctx["names"].get(appid, f"appid {appid}")
                st.write(f"- {name} (score={score:.3f})")

with tab_live:
    st.caption(
        "팀 데이터(779명)에 없는 steamid를 입력하면 Steam API로 실시간 조회해 그 자리에서 "
        "취향벡터를 계산하고 추천합니다. 프로필과 '게임 세부정보'가 공개(Public)여야 합니다."
    )

    try:
        steam_client.get_api_key()
        has_key = True
    except steam_client.SteamAPIError as e:
        has_key = False
        st.warning(str(e))

    live_steamid = st.text_input("steamid64 입력", placeholder="예: 76561197960612825", disabled=not has_key)
    lookup = st.button("조회", disabled=not has_key)

    if lookup and live_steamid:
        with st.spinner("Steam API 조회 중..."):
            try:
                owned_games, scores, n_matched, n_total = live_recommend.recommend_live(live_steamid, ctx)
            except steam_client.SteamAPIError as e:
                st.error(str(e))
                owned_games = None

        if owned_games is not None:
            st.info(f"보유 게임 {n_total}개 중 {n_matched}개가 우리 카탈로그와 매칭됨 (매칭된 게임만 취향벡터에 반영)")

            top5_played_live = owned_games.nlargest(5, "playtime_hours")
            top5_rec_live = scores.nlargest(5)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("많이 플레이한 게임")
                for row in top5_played_live.itertuples():
                    st.write(f"- {row.game_name} ({row.playtime_hours:.0f}h)")

            with col2:
                st.subheader("추천 Top-5 (Content 기반, 실시간)")
                for appid, score in top5_rec_live.items():
                    name = ctx["names"].get(appid, f"appid {appid}")
                    st.write(f"- {name} (score={score:.3f})")
