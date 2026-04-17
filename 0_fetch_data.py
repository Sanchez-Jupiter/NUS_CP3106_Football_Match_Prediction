import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from requests.exceptions import HTTPError
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = {
    39: "EPL",
    140: "LaLiga",
    135: "SerieA",
    78: "Bundesliga",
    61: "Ligue1",
}

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]  
OUT_ROOT = Path("data/raw/api_football")
SLEEP_SEC = 0.6   # 控频，避免打满限额
RETRY = 3

def _extract_api_message(resp):
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            if "errors" in payload and payload["errors"]:
                return str(payload["errors"])
            if "message" in payload:
                return str(payload["message"])
            if "response" in payload and payload["response"]:
                return str(payload["response"])
    except Exception:
        pass
    text = (resp.text or "").strip()
    return text[:500] if text else "No response body"

def api_get(endpoint, params):
    url = BASE_URL + endpoint
    for i in range(RETRY):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code in (401, 403):
                api_message = _extract_api_message(r)
                raise PermissionError(
                    f"HTTP {r.status_code} on {endpoint} params={params}. "
                    f"API says: {api_message}. "
                    "Check API_KEY in code, subscription access, and historical data rights."
                )
            if r.status_code == 429:
                # 触发限流则指数退避
                time.sleep((2 ** i) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except PermissionError:
            raise
        except HTTPError:
            if i == RETRY - 1:
                raise
            time.sleep(2 ** i)
        except Exception:
            if i == RETRY - 1:
                raise
            time.sleep(2 ** i)


def preflight_check():
    if not API_KEY:
        raise ValueError(
            "Missing API key. Fill API_KEY in this script before running."
        )
    status = api_get("/status", {})
    print("API preflight OK.")
    save_json(OUT_ROOT / "api_status.json", status)

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def fetch_fixtures(league_id, season):
    return api_get("/fixtures", {"league": league_id, "season": season})

def fetch_fixture_detail(fixture_id):
    bundle = {}
    bundle["events"] = api_get("/fixtures/events", {"fixture": fixture_id})
    bundle["statistics"] = api_get("/fixtures/statistics", {"fixture": fixture_id})
    bundle["lineups"] = api_get("/fixtures/lineups", {"fixture": fixture_id})
    bundle["players"] = api_get("/fixtures/players", {"fixture": fixture_id})
    bundle["meta"] = api_get("/fixtures", {"id": fixture_id})
    return bundle

def main():
    preflight_check()

    for league_id, league_name in LEAGUES.items():
        for season in SEASONS:
            season_dir = OUT_ROOT / f"league={league_id}_{league_name}" / f"season={season}"
            print(f"Downloading fixtures: league={league_id}, season={season}")
            try:
                fixtures_obj = fetch_fixtures(league_id, season)
            except PermissionError as e:
                season_dir.mkdir(parents=True, exist_ok=True)
                (season_dir / "_season_error.txt").write_text(str(e), encoding="utf-8")
                print(f"Skip league={league_id}, season={season}: {e}")
                continue

            save_json(season_dir / "fixtures.json", fixtures_obj)

            # 检查 API 返回的 errors 或其他异常标志
            fixtures = fixtures_obj.get("response", [])
            api_errors = fixtures_obj.get("errors", {})
            
            if api_errors:
                season_dir.mkdir(parents=True, exist_ok=True)
                err_msg = f"API returned errors: {api_errors}"
                (season_dir / "_season_error.txt").write_text(err_msg, encoding="utf-8")
                print(f"  [WARNING] {err_msg}")
            
            print(f"Total fixtures: {len(fixtures)}")
            
            if not fixtures:
                print(f"  No fixtures for league={league_id}, season={season}. "
                      f"This may be: subscription limitation, future season, or data not available.")

            for fx in tqdm(fixtures, desc=f"{league_name}-{season}"):
                fixture_id = fx["fixture"]["id"]
                fixture_dir = season_dir / f"fixture_{fixture_id}"
                done_flag = fixture_dir / "_done.flag"
                if done_flag.exists():
                    continue

                try:
                    details = fetch_fixture_detail(fixture_id)
                    for k, v in details.items():
                        save_json(fixture_dir / f"{k}.json", v)
                    done_flag.write_text("ok", encoding="utf-8")
                except Exception as e:
                    err_path = fixture_dir / "_error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(str(e), encoding="utf-8")

                time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()