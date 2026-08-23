"""
Steam Web API 실시간 조회 — 팀 데이터(779명)에 없는 새 steamid의 보유 게임을
그 자리에서 가져온다.

API 키는 반드시 환경변수 STEAM_API_KEY로 읽는다 (코드에 하드코딩 금지).
발급: https://steamcommunity.com/dev/apikey
"""

import os

import pandas as pd
import requests

OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"


class SteamAPIError(Exception):
    """Steam API 키 누락, 호출 실패, 비공개 프로필 등을 알리기 위한 예외."""


def get_api_key() -> str:
    key = os.environ.get("STEAM_API_KEY")
    if not key:
        raise SteamAPIError(
            "STEAM_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "https://steamcommunity.com/dev/apikey 에서 키를 발급받아 "
            "`export STEAM_API_KEY=...` 또는 .env 파일에 STEAM_API_KEY=... 로 설정하세요."
        )
    return key


def fetch_owned_games(steamid: str, timeout: float = 10.0) -> pd.DataFrame:
    """steamid의 보유 게임 목록을 Steam API에서 실시간으로 가져온다.

    Returns
    -------
    DataFrame[appid, game_name, playtime_hours]
    """
    params = {
        "key": get_api_key(),
        "steamid": steamid,
        "include_appinfo": True,
        "include_played_free_games": True,
        "format": "json",
    }
    try:
        resp = requests.get(OWNED_GAMES_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise SteamAPIError(f"Steam API 호출 실패: {e}") from e

    games = data.get("response", {}).get("games")
    if not games:
        raise SteamAPIError(
            "보유 게임 정보를 가져오지 못했습니다. steamid가 올바른지, "
            "프로필과 '게임 세부정보'가 공개(Public)로 설정되어 있는지 확인하세요."
        )

    rows = [
        {
            "appid": g["appid"],
            "game_name": g.get("name", "Unknown"),
            "playtime_hours": g.get("playtime_forever", 0) / 60,
        }
        for g in games
    ]
    return pd.DataFrame(rows)
