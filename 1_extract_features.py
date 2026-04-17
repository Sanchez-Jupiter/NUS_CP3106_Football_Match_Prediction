"""
Step 1 : Extracting pre-competition and in-competition features from raw JSON
Outputs two CSV files:
  - data/processed/pretrain_dataset.csv  (Pre-competition training dataset)
  - data/processed/inplay_dataset.csv    (In-competition training dataset)
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = Path("data/raw/api_football")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPLAY_CHECKPOINT_MINUTES = [10, 20, 30, 40, 45, 50, 60, 70, 80, 90]
INPLAY_RECENT_WINDOW = 10 # Number of minutes to look back for "recent" in-play events (e.g. recent goals/cards/subs)

# FixtureLoader is responsible for loading the raw JSON data for each fixture 
# and extracting the relevant information into a structured format.
class FixtureLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fixtures = []
    
    def load_all(self):
        """Load all downloaded fixture data"""
        print("Loading all fixtures...")
        
        for league_dir in self.data_dir.glob("league=*"):# self.data_dir.glob 是用来查找符合特定模式的文件或目录的函数，这里它会返回一个生成器，生成所有以 "league=" 开头的目录路径。
            league_name = league_dir.name.split("_")[1]
            for season_dir in league_dir.glob("season=*"):
                season = int(season_dir.name.split("=")[1])
                for fixture_dir in season_dir.glob("fixture_*"):
                    fixture = self._load_fixture(fixture_dir, league_name, season)
                    if fixture:
                        self.fixtures.append(fixture)
        
        print(f"Total fixtures loaded: {len(self.fixtures)}")
        return self.fixtures
    
    def _load_fixture(self, fixture_dir, league, season):
        try:
            meta_full = self._load_json(fixture_dir / "meta.json")
            if not meta_full or not isinstance(meta_full, dict):
                return None
            # meta.json may contain a "response" list, which holds the actual match data
            response_list = meta_full.get("response", [])
            if not response_list or len(response_list) == 0:
                return None
            
            fx_data = response_list[0]
            score_data = fx_data.get("score", {}) or {} # score_data may be None or missing, so we default to an empty dict to avoid errors
            fulltime_data = score_data.get("fulltime", {}) or {}
            halftime_data = score_data.get("halftime", {}) or {}
            ft_home = fulltime_data.get("home")
            ft_away = fulltime_data.get("away")

            # Skip fixtures without a final score. These are usually postponed,
            # cancelled, or otherwise incomplete and cannot be used for labels.
            if ft_home is None or ft_away is None:
                return None

            events_full = self._load_json(fixture_dir / "events.json")
            events_response = events_full.get("response", []) if events_full and isinstance(events_full, dict) else []
            # Load lineups
            lineups_full = self._load_json(fixture_dir / "lineups.json")
            lineups_response = lineups_full.get("response", []) if lineups_full and isinstance(lineups_full, dict) else []
            players_full = self._load_json(fixture_dir / "players.json")
            players_response = players_full.get("response", []) if players_full and isinstance(players_full, dict) else []
            
            return {
                "fixture_id": fx_data["fixture"]["id"],
                "league": league,
                "season": season,
                "date": fx_data["fixture"]["date"],
                "timestamp": fx_data["fixture"]["timestamp"],
                "home_team_id": fx_data["teams"]["home"]["id"],
                "home_team": fx_data["teams"]["home"]["name"],
                "away_team_id": fx_data["teams"]["away"]["id"],
                "away_team": fx_data["teams"]["away"]["name"],
                "venue": fx_data["fixture"]["venue"]["name"],
                "referee": fx_data["fixture"]["referee"],
                "round": fx_data.get("league", {}).get("round", ""),
                "ft_home": ft_home,
                "ft_away": ft_away,
                "ht_home": halftime_data.get("home"),
                "ht_away": halftime_data.get("away"),
                "events": events_response,
                "lineups": lineups_response,
                "players": players_response,
                "fixture_dir": fixture_dir,
            }
        except Exception as e:
            print(f"Error loading fixture {fixture_dir}: {e}")
            return None
    
    @staticmethod 
    def _load_json(path):
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            return None


# TeamStats is responsible for processing the loaded fixtures 
# and extracting various statistics and features for each team, as well as head-to-head features and key player features. 
# It organizes the historical data in a way that allows for easy retrieval of features before a given match timestamp.
class TeamStats:
    def __init__(self, fixtures):
        self.fixtures = sorted(fixtures, key=lambda x: x["timestamp"])
        # lambda x: x["timestamp"] 是一个匿名函数，用于从每个元素 x 中提取 "timestamp" 键的值。
        # 这个函数被用作 sorted() 函数的 key 参数，以便根据时间戳对 fixtures 列表中的字典进行排序。
        self.team_history = defaultdict(list) 
        self.h2h_history = defaultdict(list) # Home and away match history for each pair of teams
        self.team_player_history = defaultdict(lambda: defaultdict(list))
        self.build_history()

    # _summarize_match_events is a static method that takes a list of events from a match and a team ID, 
    # and summarizes the key statistics for that team in the match.
    @staticmethod
    def _summarize_match_events(events, team_id):
        team_events = [e for e in events if e.get("team", {}).get("id") == team_id]
        goals = [e for e in team_events if e.get("type") == "Goal"]
        cards = [e for e in team_events if e.get("type") == "Card"]
        subs = [e for e in team_events if e.get("type") == "subst"]

        yellow = sum(1 for e in cards if "yellow" in str(e.get("detail", "")).lower())
        red = sum(1 for e in cards if "red" in str(e.get("detail", "")).lower())

        first_half_goals = 0
        second_half_goals = 0
        first_goal_min = None
        # Calculate the number of goals in the first and second halves, and the time of the first goal
        # elapsed is an integer representing the minute of the match (may exceed 90 minutes, including stoppage time)
        for e in goals:
            elapsed = e.get("time", {}).get("elapsed", 0) or 0
            if elapsed <= 45:
                first_half_goals += 1
            else:
                second_half_goals += 1
            if first_goal_min is None or elapsed < first_goal_min:
                first_goal_min = elapsed

        return {
            "yellow": yellow,
            "red": red,
            "subs": len(subs),
            "first_half_goals": first_half_goals,
            "second_half_goals": second_half_goals,
            "first_goal_min": first_goal_min,
        }

    # _safe_float是一个静态方法，用于安全地将输入值转换为浮点数。
    # 如果转换失败或输入值为None，则返回默认值（默认为0.0）。
    # 这个方法在处理可能包含非数值数据的统计信息时非常有用，可以避免程序因类型错误而崩溃。
    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    # _extract_round_no is a static method that extracts the round number from a given round text string.
    @staticmethod
    def _extract_round_no(round_text):
        if not round_text:
            return 0
        m = re.search(r"(\d+)", str(round_text))
        # re.search(r"(\d+)", str(round_text)) 是一个正则表达式搜索，
        # 用于从 round_text 字符串中提取第一个连续的数字序列。
        # 这个数字通常表示比赛的轮次（round number）。
        # 如果找到匹配项，m.group(1) 将返回这个数字；
        # 如果没有找到匹配项，则返回 None，此时函数会返回默认值 0。
        return int(m.group(1)) if m else 0

    # _lineup_player_ids is a static method that takes the lineup information from a fixture and a team ID,
    # and returns a set of player IDs for the starting XI of that team.
    @staticmethod
    def _lineup_player_ids(fx, team_id):
        ids = set()
        for team_data in fx.get("lineups", []) or []:
            # fx.get("lineups", []) or [] 是一种安全的访问方式，用于获取 fx 字典中的 "lineups" 键对应的值。
            if team_data.get("team", {}).get("id") != team_id:
                continue
            for player_item in team_data.get("startXI", []) or []:
                pid = player_item.get("player", {}).get("id")
                if pid is not None:
                    ids.add(pid)
        return ids
    #  _iter_team_player_entries is a static method that iterates through the player entries for a given team in a fixture,
    #  yielding the player information and their statistics.
    @staticmethod
    def _iter_team_player_entries(fx, team_id):
        for team_block in fx.get("players", []) or []:
            if team_block.get("team", {}).get("id") != team_id:
                continue
            for player_block in team_block.get("players", []) or []:
                player_info = player_block.get("player", {})
                stats_list = player_block.get("statistics", []) or [] 
                stats = stats_list[0] if stats_list else {}
                yield player_info, stats 

    # build_history processes all the fixtures and builds the historical data structures for team performance, head-to-head matchups, and player statistics.
    def build_history(self):
        for fx in self.fixtures:
            if fx.get("ft_home") is None or fx.get("ft_away") is None:
                continue

            home = fx["home_team"]
            away = fx["away_team"]
            events = fx.get("events", [])

            home_event_stats = self._summarize_match_events(events, fx["home_team_id"])
            away_event_stats = self._summarize_match_events(events, fx["away_team_id"])
            
            result = self._get_result(fx["ft_home"], fx["ft_away"])
            
            self.team_history[home].append({
                "timestamp": fx["timestamp"],
                "fixture_id": fx["fixture_id"],
                "league": fx["league"],
                "season": fx["season"],
                "round_no": self._extract_round_no(fx.get("round", "")),
                "is_home": True,
                "opponent": away,
                "goals_for": fx["ft_home"],
                "goals_against": fx["ft_away"],
                "result": result,
                "points": 3 if result == "W" else (1 if result == "D" else 0),
                "clean_sheet": 1 if fx["ft_away"] == 0 else 0,
                "failed_to_score": 1 if fx["ft_home"] == 0 else 0,
                "yellow": home_event_stats["yellow"],
                "red": home_event_stats["red"],
                "subs": home_event_stats["subs"],
                "first_half_goals": home_event_stats["first_half_goals"],
                "second_half_goals": home_event_stats["second_half_goals"],
                "first_half_against": away_event_stats["first_half_goals"],
                "second_half_against": away_event_stats["second_half_goals"],
            })
            
            result_away = "W" if result == "L" else ("L" if result == "W" else "D")
            self.team_history[away].append({
                "timestamp": fx["timestamp"],
                "fixture_id": fx["fixture_id"],
                "league": fx["league"],
                "season": fx["season"],
                "round_no": self._extract_round_no(fx.get("round", "")),
                "is_home": False,
                "opponent": home,
                "goals_for": fx["ft_away"],
                "goals_against": fx["ft_home"],
                "result": result_away,
                "points": 3 if result_away == "W" else (1 if result_away == "D" else 0),
                "clean_sheet": 1 if fx["ft_home"] == 0 else 0,
                "failed_to_score": 1 if fx["ft_away"] == 0 else 0,
                "yellow": away_event_stats["yellow"],
                "red": away_event_stats["red"],
                "subs": away_event_stats["subs"],
                "first_half_goals": away_event_stats["first_half_goals"],
                "second_half_goals": away_event_stats["second_half_goals"],
                "first_half_against": home_event_stats["first_half_goals"],
                "second_half_against": home_event_stats["second_half_goals"],
            })

            pair_key = tuple(sorted([home, away]))
            self.h2h_history[pair_key].append({
                "timestamp": fx["timestamp"],
                "home": home,
                "away": away,
                "home_result": result,
                "home_goal_diff": fx["ft_home"] - fx["ft_away"],
            })

            for team_name, team_id in ((home, fx["home_team_id"]), (away, fx["away_team_id"])):
                for player_info, stats in self._iter_team_player_entries(fx, team_id):
                    pid = player_info.get("id")
                    if pid is None:
                        continue
                    games = stats.get("games", {}) or {}
                    goals = stats.get("goals", {}) or {}
                    cards = stats.get("cards", {}) or {}
                    self.team_player_history[team_name][pid].append({
                        "timestamp": fx["timestamp"],
                        "minutes": self._safe_float(games.get("minutes"), 0.0),
                        "rating": self._safe_float(games.get("rating"), 0.0),
                        "goals": self._safe_float(goals.get("total"), 0.0),
                        "assists": self._safe_float(goals.get("assists"), 0.0),
                        "yellow": self._safe_float(cards.get("yellow"), 0.0),
                        "red": self._safe_float(cards.get("red"), 0.0),
                    })
    
    # get_features_before retrieves the aggregated performance features for a given team before a specified timestamp, 
    # looking back at a certain number of recent games.
    def get_features_before(self, team, timestamp, lookback_games=5):
        history = [g for g in self.team_history[team] if g["timestamp"] < timestamp]
        
        if not history:
            return self._empty_features()
        
        recent = history[-lookback_games:]
        
        wins = sum(1 for g in recent if g["result"] == "W")
        draws = sum(1 for g in recent if g["result"] == "D")
        losses = sum(1 for g in recent if g["result"] == "L")
        
        goals_for = sum(g["goals_for"] for g in recent)
        goals_against = sum(g["goals_against"] for g in recent)
        points = sum(g["points"] for g in recent)
        clean_sheets = sum(g["clean_sheet"] for g in recent)
        failed_to_score = sum(g["failed_to_score"] for g in recent)
        yellows = sum(g["yellow"] for g in recent)
        reds = sum(g["red"] for g in recent)
        subs = sum(g["subs"] for g in recent)

        first_half_gf = sum(g["first_half_goals"] for g in recent)
        second_half_gf = sum(g["second_half_goals"] for g in recent)
        first_half_ga = sum(g["first_half_against"] for g in recent)
        second_half_ga = sum(g["second_half_against"] for g in recent)

        denom = len(recent)
        
        return {
            "games_played": len(history),
            "recent_wins": wins,
            "recent_draws": draws,
            "recent_losses": losses,
            "recent_gf": goals_for,
            "recent_ga": goals_against,
            "recent_gd": goals_for - goals_against,
            "win_rate": wins / denom,
            "avg_gf": goals_for / denom,
            "avg_ga": goals_against / denom,
            "points_per_game": points / denom,
            "clean_sheet_rate": clean_sheets / denom,
            "failed_to_score_rate": failed_to_score / denom,
            "avg_yellow": yellows / denom,
            "avg_red": reds / denom,
            "avg_subs": subs / denom,
            "avg_first_half_gf": first_half_gf / denom,
            "avg_second_half_gf": second_half_gf / denom,
            "avg_first_half_ga": first_half_ga / denom,
            "avg_second_half_ga": second_half_ga / denom,
        }

    # get_h2h_features_before retrieves the head-to-head features for a pair of teams before a specified timestamp,
    # looking back at a certain number of recent head-to-head games. It calculates the number of games, win/draw rates, 
    # and average goal difference for the home team in those matchups.
    def get_h2h_features_before(self, home_team, away_team, timestamp, lookback_games=5):
        pair_key = tuple(sorted([home_team, away_team]))
        history = [g for g in self.h2h_history[pair_key] if g["timestamp"] < timestamp]

        if not history:
            return {
                "h2h_games": 0,
                "h2h_home_win_rate": 0.0,
                "h2h_draw_rate": 0.0,
                "h2h_home_goal_diff_avg": 0.0,
            }

        recent = history[-lookback_games:]
        home_wins = 0
        draws = 0
        goal_diff_sum = 0.0

        for g in recent:
            if g["home"] == home_team and g["away"] == away_team:
                result_for_home = g["home_result"]
                gd_for_home = g["home_goal_diff"]
            else:
                if g["home_result"] == "W":
                    result_for_home = "L"
                elif g["home_result"] == "L":
                    result_for_home = "W"
                else:
                    result_for_home = "D"
                gd_for_home = -g["home_goal_diff"]

            if result_for_home == "W":
                home_wins += 1
            elif result_for_home == "D":
                draws += 1
            goal_diff_sum += gd_for_home

        denom = len(recent)
        return {
            "h2h_games": len(history),
            "h2h_home_win_rate": home_wins / denom,
            "h2h_draw_rate": draws / denom,
            "h2h_home_goal_diff_avg": goal_diff_sum / denom,
        }

    # get_schedule_features_before retrieves the schedule-related features for a given team before a specified timestamp,
    # such as the number of days since the last match and the number of matches played in the last 7 and 14 days.
    def get_schedule_features_before(self, team, timestamp):
        history = [g for g in self.team_history[team] if g["timestamp"] < timestamp]
        if not history:
            return {
                "days_since_last_match": 14.0,
                "matches_last_7d": 0,
                "matches_last_14d": 0,
            }

        last_ts = history[-1]["timestamp"]
        days_since_last = max(0.0, (timestamp - last_ts) / 86400.0)
        matches_7d = sum(1 for g in history if 0 < (timestamp - g["timestamp"]) <= 7 * 86400)
        matches_14d = sum(1 for g in history if 0 < (timestamp - g["timestamp"]) <= 14 * 86400)

        return {
            "days_since_last_match": days_since_last,
            "matches_last_7d": matches_7d,
            "matches_last_14d": matches_14d,
        }

    # get_match_importance_features calculates the importance of a match for the home and away teams based on their current league standings 
    # and the progress of the season.
    def get_match_importance_features(self, fx):
        league = fx["league"]
        season = fx["season"]
        ts = fx["timestamp"]
        home = fx["home_team"]
        away = fx["away_team"]
        round_no = self._extract_round_no(fx.get("round", ""))

        league_hist = [
            x for x in self.fixtures
            if x["league"] == league and x["season"] == season and x["timestamp"] < ts
        ]

        if not league_hist:
            return {
                "round_no": round_no,
                "season_progress": 0.0,
                "h_rank": 10.0,
                "a_rank": 10.0,
                "h_gap_top": 0.0,
                "a_gap_top": 0.0,
                "h_gap_top4": 0.0,
                "a_gap_top4": 0.0,
                "h_gap_safety": 0.0,
                "a_gap_safety": 0.0,
                "importance_sum": 0.0,
                "importance_diff": 0.0,
            }

        table = defaultdict(lambda: {"pts": 0})
        for m in league_hist:
            r = self._get_result(m["ft_home"], m["ft_away"])
            h_pts = 3 if r == "W" else (1 if r == "D" else 0)
            a_pts = 3 if r == "L" else (1 if r == "D" else 0)
            table[m["home_team"]]["pts"] += h_pts
            table[m["away_team"]]["pts"] += a_pts

        ranked = sorted(table.items(), key=lambda kv: kv[1]["pts"], reverse=True)
        ranks = {team: i + 1 for i, (team, _) in enumerate(ranked)}

        top_pts = ranked[0][1]["pts"] if ranked else 0
        top4_pts = ranked[3][1]["pts"] if len(ranked) >= 4 else top_pts
        safe_pts = ranked[16][1]["pts"] if len(ranked) >= 17 else 0

        h_pts = table[home]["pts"]
        a_pts = table[away]["pts"]
        h_rank = ranks.get(home, float(len(ranked) + 1))
        a_rank = ranks.get(away, float(len(ranked) + 1))

        h_gap_top = top_pts - h_pts
        a_gap_top = top_pts - a_pts
        h_gap_top4 = top4_pts - h_pts
        a_gap_top4 = top4_pts - a_pts
        h_gap_safety = h_pts - safe_pts
        a_gap_safety = a_pts - safe_pts

        # Higher = match is more important for season objective (title/top4/relegation)
        h_importance = (
            np.exp(-max(0.0, h_gap_top) / 6.0)
            + np.exp(-abs(h_gap_top4) / 4.0)
            + np.exp(-abs(h_gap_safety) / 4.0)
        )
        a_importance = (
            np.exp(-max(0.0, a_gap_top) / 6.0)
            + np.exp(-abs(a_gap_top4) / 4.0)
            + np.exp(-abs(a_gap_safety) / 4.0)
        )

        return {
            "round_no": round_no,
            "season_progress": round_no / 38.0 if round_no > 0 else 0.0,
            "h_rank": float(h_rank),
            "a_rank": float(a_rank),
            "h_gap_top": float(h_gap_top),
            "a_gap_top": float(a_gap_top),
            "h_gap_top4": float(h_gap_top4),
            "a_gap_top4": float(a_gap_top4),
            "h_gap_safety": float(h_gap_safety),
            "a_gap_safety": float(a_gap_safety),
            "importance_sum": float(h_importance + a_importance),
            "importance_diff": float(h_importance - a_importance),
        }


    # get_key_player_features retrieves the features related to key players for a given team before a specified timestamp.
    def get_key_player_features(self, team, team_id, fx, timestamp, top_k=3):
        player_hist = self.team_player_history.get(team, {})
        if not player_hist:
            return {
                "key_players_started": 0,
                "key_players_absent": top_k,
                "key_players_form_avg_rating": 0.0,
                "key_players_form_avg_contrib": 0.0,
                "starting11_avg_minutes_7d": 0.0,
                "starting11_avg_matches_7d": 0.0,
            }

        lineup_ids = self._lineup_player_ids(fx, team_id)

        scored = []
        for pid, recs in player_hist.items():
            hist = [r for r in recs if r["timestamp"] < timestamp]
            if not hist:
                continue
            recent = hist[-5:]
            contrib = np.mean([
                r["goals"] * 4.0 + r["assists"] * 3.0 + max(0.0, r["rating"] - 6.0)
                for r in recent
            ])
            avg_rating = np.mean([r["rating"] for r in recent]) if recent else 0.0
            scored.append((pid, contrib, avg_rating))

        if not scored:
            return {
                "key_players_started": 0,
                "key_players_absent": top_k,
                "key_players_form_avg_rating": 0.0,
                "key_players_form_avg_contrib": 0.0,
                "starting11_avg_minutes_7d": 0.0,
                "starting11_avg_matches_7d": 0.0,
            }

        scored.sort(key=lambda x: x[1], reverse=True)
        key_players = scored[:top_k]
        key_ids = {x[0] for x in key_players}
        started = len(key_ids.intersection(lineup_ids))

        avg_rating = float(np.mean([x[2] for x in key_players])) if key_players else 0.0
        avg_contrib = float(np.mean([x[1] for x in key_players])) if key_players else 0.0

        # Player schedule density of current starting XI
        minutes_7d = []
        matches_7d = []
        for pid in lineup_ids:
            recs = [r for r in player_hist.get(pid, []) if 0 < (timestamp - r["timestamp"]) <= 7 * 86400]
            minutes_7d.append(sum(r["minutes"] for r in recs))
            matches_7d.append(len(recs))

        return {
            "key_players_started": started,
            "key_players_absent": max(0, top_k - started),
            "key_players_form_avg_rating": avg_rating,
            "key_players_form_avg_contrib": avg_contrib,
            "starting11_avg_minutes_7d": float(np.mean(minutes_7d)) if minutes_7d else 0.0,
            "starting11_avg_matches_7d": float(np.mean(matches_7d)) if matches_7d else 0.0,
        }
    
    @staticmethod
    def _get_result(home_goals, away_goals):
        if home_goals > away_goals:
            return "W"
        elif home_goals < away_goals:
            return "L"
        else:
            return "D"
    
    # _empty_features returns a dictionary of default feature values for a team with no historical data. 
    # This is used to handle cases where a team has not played any matches before the given timestamp, 
    # ensuring that the feature extraction process can still proceed without errors.
    @staticmethod
    def _empty_features():
        return {
            "games_played": 0,
            "recent_wins": 0,
            "recent_draws": 0,
            "recent_losses": 0,
            "recent_gf": 0,
            "recent_ga": 0,
            "recent_gd": 0,
            "win_rate": 0.0,
            "avg_gf": 0.0,
            "avg_ga": 0.0,
            "points_per_game": 0.0,
            "clean_sheet_rate": 0.0,
            "failed_to_score_rate": 0.0,
            "avg_yellow": 0.0,
            "avg_red": 0.0,
            "avg_subs": 0.0,
            "avg_first_half_gf": 0.0,
            "avg_second_half_gf": 0.0,
            "avg_first_half_ga": 0.0,
            "avg_second_half_ga": 0.0,
        }

# InPlayFeatures is responsible for extracting in-play features from the match events at specific time checkpoints during the match.
class InPlayFeatures:
    SEGMENTS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45), (46, 50), (51, 60), (61, 70), (71, 80), (81, 90)]

    # _safe_elapsed is a static method that safely retrieves the elapsed time (in minutes) from an event dictionary.
    @staticmethod
    def _safe_elapsed(event):
        return int(event.get("time", {}).get("elapsed", 0) or 0)

    @staticmethod
    def extract_events_at_minute(events, minute):
        """Extract events that occurred up to a specific minute."""
        filtered = [e for e in events if InPlayFeatures._safe_elapsed(e) <= minute]
        return sorted(filtered, key=InPlayFeatures._safe_elapsed)

    # extract_events_between is a static method that extracts events that occurred between two specified minutes in the match.
    @staticmethod
    def extract_events_between(events, start_min, end_min):
        return [
            e
            for e in events
            if start_min < InPlayFeatures._safe_elapsed(e) <= end_min
        ]
    # count_events is a static method that counts the number of events of a specific type and/or for a specific team from a list of events.
    @staticmethod
    def count_events(events, event_type=None, team_id=None):
        """Count events of a specific type and/or team."""
        result = []
        for e in events:
            if event_type and e.get("type") != event_type:
                continue
            if team_id and e.get("team", {}).get("id") != team_id:
                continue
            result.append(e)
        return result
    # count_cards is a static method that counts the number of cards of a specific color (yellow or red) for a given team from a list of events.
    @staticmethod
    def count_cards(events, team_id, color):
        """Count cards of a specific color for a team."""
        cards = InPlayFeatures.count_events(events, "Card", team_id)
        color_lower = color.lower()
        return sum(1 for e in cards if color_lower in str(e.get("detail", "")).lower())

    # Define an event impact scoring function. The score is positive if it 
    # favors the home team and negative if it favors the away team.
    @staticmethod
    def event_impact(event, home_team_id, away_team_id):
        etype = event.get("type", "")
        detail = str(event.get("detail", "")).lower()
        team_id = event.get("team", {}).get("id")

        sign = 1.0 if team_id == home_team_id else (-1.0 if team_id == away_team_id else 0.0)
        if sign == 0.0:
            return 0.0

        if etype == "Goal":
            base = 3.0
            if "penalty" in detail:
                base += 0.5
            if "own goal" in detail:
                base += 0.5
            return sign * base
        if etype == "Card":
            if "red" in detail:
                return sign * (-2.5)
            if "yellow" in detail:
                return sign * (-0.8)
        if etype == "subst":
            return sign * 0.2
        if etype == "Var":
            return sign * 0.6
        return 0.0

    # Calculate lead changes and equalizers up to the current minute
    @staticmethod
    def _lead_change_features(events_timeline, home_team_id, away_team_id):
        score_h = 0
        score_a = 0
        lead_changes = 0
        equalizers = 0
        leader = 0
        first_goal_team = 0
        for e in events_timeline:
            if e.get("type") != "Goal":
                continue
            team_id = e.get("team", {}).get("id")
            if team_id == home_team_id:
                score_h += 1
                if first_goal_team == 0:
                    first_goal_team = 1
            elif team_id == away_team_id:
                score_a += 1
                if first_goal_team == 0:
                    first_goal_team = -1

            new_leader = 1 if score_h > score_a else (-1 if score_a > score_h else 0)
            if new_leader == 0 and leader != 0:
                equalizers += 1
            if new_leader != 0 and leader != 0 and new_leader != leader:
                lead_changes += 1
            leader = new_leader

        return {
            "first_goal_team": first_goal_team,
            "lead_changes": lead_changes,
            "equalizers": equalizers,
        }
    
    # get_inplay_features is a static method that extracts various in-play features from the match events at a given minute.
    @staticmethod
    def get_inplay_features(fx, minute, home_team_id, away_team_id, recent_window=INPLAY_RECENT_WINDOW):
        """In-play features at a given minute."""
        events = fx.get("events", [])
        events_at_min = InPlayFeatures.extract_events_at_minute(events, minute)
        events_recent = InPlayFeatures.extract_events_between(events, max(0, minute - recent_window), minute)
        
        # Goals
        goals_home = len(InPlayFeatures.count_events(events_at_min, "Goal", home_team_id))
        goals_away = len(InPlayFeatures.count_events(events_at_min, "Goal", away_team_id))

        goals_home_recent = len(InPlayFeatures.count_events(events_recent, "Goal", home_team_id))
        goals_away_recent = len(InPlayFeatures.count_events(events_recent, "Goal", away_team_id))
        
        # Yellow/Red cards
        yellow_home = InPlayFeatures.count_cards(events_at_min, home_team_id, "yellow")
        yellow_away = InPlayFeatures.count_cards(events_at_min, away_team_id, "yellow")
        yellow_home_recent = InPlayFeatures.count_cards(events_recent, home_team_id, "yellow")
        yellow_away_recent = InPlayFeatures.count_cards(events_recent, away_team_id, "yellow")
        
        red_home = InPlayFeatures.count_cards(events_at_min, home_team_id, "red")
        red_away = InPlayFeatures.count_cards(events_at_min, away_team_id, "red")
        red_home_recent = InPlayFeatures.count_cards(events_recent, home_team_id, "red")
        red_away_recent = InPlayFeatures.count_cards(events_recent, away_team_id, "red")
        
        # Substitutions
        subs_home = len(InPlayFeatures.count_events(events_at_min, "subst", home_team_id))
        subs_away = len(InPlayFeatures.count_events(events_at_min, "subst", away_team_id))
        subs_home_recent = len(InPlayFeatures.count_events(events_recent, "subst", home_team_id))
        subs_away_recent = len(InPlayFeatures.count_events(events_recent, "subst", away_team_id))

        var_home = len(InPlayFeatures.count_events(events_at_min, "Var", home_team_id))
        var_away = len(InPlayFeatures.count_events(events_at_min, "Var", away_team_id))

        # Time-related features
        goal_events_home = InPlayFeatures.count_events(events_at_min, "Goal", home_team_id)
        goal_events_away = InPlayFeatures.count_events(events_at_min, "Goal", away_team_id)
        last_goal_home_min = max([InPlayFeatures._safe_elapsed(e) for e in goal_events_home], default=0)
        last_goal_away_min = max([InPlayFeatures._safe_elapsed(e) for e in goal_events_away], default=0)

        min_since_last_goal_home = minute - last_goal_home_min if last_goal_home_min > 0 else 999
        min_since_last_goal_away = minute - last_goal_away_min if last_goal_away_min > 0 else 999

        # Event impact scores
        impact_total = sum(InPlayFeatures.event_impact(e, home_team_id, away_team_id) for e in events_at_min)
        impact_recent = sum(InPlayFeatures.event_impact(e, home_team_id, away_team_id) for e in events_recent)

        lead_feats = InPlayFeatures._lead_change_features(events_at_min, home_team_id, away_team_id)

        # Segment features: goal difference, red card difference, impact score for each 15-minute segment
        segment_features = {}
        for seg_start, seg_end in InPlayFeatures.SEGMENTS:
            seg_name = f"{seg_start}_{seg_end}"
            seg_events = InPlayFeatures.extract_events_between(events, seg_start - 1, min(seg_end, minute))
            seg_goal_h = len(InPlayFeatures.count_events(seg_events, "Goal", home_team_id))
            seg_goal_a = len(InPlayFeatures.count_events(seg_events, "Goal", away_team_id))
            seg_red_h = InPlayFeatures.count_cards(seg_events, home_team_id, "red")
            seg_red_a = InPlayFeatures.count_cards(seg_events, away_team_id, "red")
            seg_impact = sum(InPlayFeatures.event_impact(e, home_team_id, away_team_id) for e in seg_events)

            segment_features[f"seg_goal_diff_{seg_name}"] = seg_goal_h - seg_goal_a
            segment_features[f"seg_red_diff_{seg_name}"] = seg_red_h - seg_red_a
            segment_features[f"seg_impact_{seg_name}"] = seg_impact
        
        return {
            "minute": minute,
            "minute_ratio": minute / 90.0,
            "goals_home": goals_home,
            "goals_away": goals_away,
            "goal_diff": goals_home - goals_away,
            f"goals_home_recent{recent_window}": goals_home_recent,
            f"goals_away_recent{recent_window}": goals_away_recent,
            f"goal_diff_recent{recent_window}": goals_home_recent - goals_away_recent,
            "yellow_home": yellow_home,
            "yellow_away": yellow_away,
            f"yellow_home_recent{recent_window}": yellow_home_recent,
            f"yellow_away_recent{recent_window}": yellow_away_recent,
            "red_home": red_home,
            "red_away": red_away,
            f"red_home_recent{recent_window}": red_home_recent,
            f"red_away_recent{recent_window}": red_away_recent,
            "subs_home": subs_home,
            "subs_away": subs_away,
            f"subs_home_recent{recent_window}": subs_home_recent,
            f"subs_away_recent{recent_window}": subs_away_recent,
            "var_home": var_home,
            "var_away": var_away,
            "min_since_last_goal_home": min_since_last_goal_home,
            "min_since_last_goal_away": min_since_last_goal_away,
            "impact_score_total": impact_total,
            f"impact_score_recent{recent_window}": impact_recent,
            **lead_feats,
            **segment_features,
        }

def get_label_outcome(ft_home, ft_away):
    """Final outcome label (H/D/A)"""
    if ft_home > ft_away:
        return "H"
    elif ft_away > ft_home:
        return "A"
    else:
        return "D"

# get_inplay_label determines the in-play label (H/D/A) based on the current scoreline at a given minute, 
# considering the remaining goals needed for each team to reach the final score. It compares the number of goals
def get_inplay_label(ft_home, ft_away, current_goals_home, current_goals_away):
    """In-play label from the current minute to the end of the match"""
    remaining_goals_h = ft_home - current_goals_home
    remaining_goals_a = ft_away - current_goals_away
    
    if remaining_goals_h > remaining_goals_a:
        return "H"
    elif remaining_goals_a > remaining_goals_h:
        return "A"
    else:
        return "D"

# generate_datasets is a function that generates both pre-match and in-play datasets by processing the fixtures and 
# extracting relevant features using the TeamStats class.
def generate_datasets(fixtures, team_stats):
    """Generate pre-match and in-play datasets"""
    
    print("\n" + "="*60)
    print("Generating pre-match dataset...")
    print("="*60)
    
    pretrain_rows = []
    inplay_rows = []
    
    for fx in tqdm(fixtures, desc="Processing fixtures"):
        # ===== Pre-match data =====
        home_features = team_stats.get_features_before(fx["home_team"], fx["timestamp"])
        away_features = team_stats.get_features_before(fx["away_team"], fx["timestamp"])
        h2h_features = team_stats.get_h2h_features_before(fx["home_team"], fx["away_team"], fx["timestamp"])
        h_sched = team_stats.get_schedule_features_before(fx["home_team"], fx["timestamp"])
        a_sched = team_stats.get_schedule_features_before(fx["away_team"], fx["timestamp"])
        imp_feats = team_stats.get_match_importance_features(fx)
        h_key = team_stats.get_key_player_features(
            fx["home_team"], fx["home_team_id"], fx, fx["timestamp"], top_k=3
        )
        a_key = team_stats.get_key_player_features(
            fx["away_team"], fx["away_team_id"], fx, fx["timestamp"], top_k=3
        )
        
        # Determine home and away formations
        home_formation = "0"
        away_formation = "0"
        if fx["lineups"] and isinstance(fx["lineups"], list):
            for team_data in fx["lineups"]:
                if team_data.get("team", {}).get("id") == fx["home_team_id"]:# 检查当前队伍是否为主队
                    home_formation = team_data.get("formation", "0")# 获取主队的阵型，如果没有提供则默认为 "0"
                elif team_data.get("team", {}).get("id") == fx["away_team_id"]:# 检查当前队伍是否为客队
                    away_formation = team_data.get("formation", "0")# 获取客队的阵型，如果没有提供则默认为 "0"
        
        pretrain_row = {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"],
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "home_formation": home_formation,
            "away_formation": away_formation,
            # Home team features
            "h_games_played": home_features["games_played"],
            "h_recent_wins": home_features["recent_wins"],
            "h_recent_draws": home_features["recent_draws"],
            "h_recent_losses": home_features["recent_losses"],
            "h_recent_gf": home_features["recent_gf"],
            "h_recent_ga": home_features["recent_ga"],
            "h_recent_gd": home_features["recent_gd"],
            "h_win_rate": home_features["win_rate"],
            "h_avg_gf": home_features["avg_gf"],
            "h_avg_ga": home_features["avg_ga"],
            # Away team features
            "a_games_played": away_features["games_played"],
            "a_recent_wins": away_features["recent_wins"],
            "a_recent_draws": away_features["recent_draws"],
            "a_recent_losses": away_features["recent_losses"],
            "a_recent_gf": away_features["recent_gf"],
            "a_recent_ga": away_features["recent_ga"],
            "a_recent_gd": away_features["recent_gd"],
            "a_win_rate": away_features["win_rate"],
            "a_avg_gf": away_features["avg_gf"],
            "a_avg_ga": away_features["avg_ga"],
            "h_points_per_game": home_features["points_per_game"],
            "a_points_per_game": away_features["points_per_game"],
            "h_clean_sheet_rate": home_features["clean_sheet_rate"],
            "a_clean_sheet_rate": away_features["clean_sheet_rate"],
            "h_failed_to_score_rate": home_features["failed_to_score_rate"],
            "a_failed_to_score_rate": away_features["failed_to_score_rate"],
            "h_avg_first_half_gf": home_features["avg_first_half_gf"],
            "a_avg_first_half_gf": away_features["avg_first_half_gf"],
            "h_avg_second_half_gf": home_features["avg_second_half_gf"],
            "a_avg_second_half_gf": away_features["avg_second_half_gf"],
            "h_avg_yellow": home_features["avg_yellow"],
            "a_avg_yellow": away_features["avg_yellow"],
            "h_avg_red": home_features["avg_red"],
            "a_avg_red": away_features["avg_red"],
            "h2h_games": h2h_features["h2h_games"],
            "h2h_home_win_rate": h2h_features["h2h_home_win_rate"],
            "h2h_draw_rate": h2h_features["h2h_draw_rate"],
            "h2h_home_goal_diff_avg": h2h_features["h2h_home_goal_diff_avg"],
            # Team schedule density
            "h_days_since_last_match": h_sched["days_since_last_match"],
            "a_days_since_last_match": a_sched["days_since_last_match"],
            "h_matches_last_7d": h_sched["matches_last_7d"],
            "a_matches_last_7d": a_sched["matches_last_7d"],
            "h_matches_last_14d": h_sched["matches_last_14d"],
            "a_matches_last_14d": a_sched["matches_last_14d"],
            # Match importance and standings context
            "round_no": imp_feats["round_no"],
            "season_progress": imp_feats["season_progress"],
            "h_rank": imp_feats["h_rank"],
            "a_rank": imp_feats["a_rank"],
            "h_gap_top": imp_feats["h_gap_top"],
            "a_gap_top": imp_feats["a_gap_top"],
            "h_gap_top4": imp_feats["h_gap_top4"],
            "a_gap_top4": imp_feats["a_gap_top4"],
            "h_gap_safety": imp_feats["h_gap_safety"],
            "a_gap_safety": imp_feats["a_gap_safety"],
            "importance_sum": imp_feats["importance_sum"],
            "importance_diff": imp_feats["importance_diff"],
            # Key-player form / availability / player schedule density
            "h_key_players_started": h_key["key_players_started"],
            "a_key_players_started": a_key["key_players_started"],
            "h_key_players_absent": h_key["key_players_absent"],
            "a_key_players_absent": a_key["key_players_absent"],
            "h_key_players_form_avg_rating": h_key["key_players_form_avg_rating"],
            "a_key_players_form_avg_rating": a_key["key_players_form_avg_rating"],
            "h_key_players_form_avg_contrib": h_key["key_players_form_avg_contrib"],
            "a_key_players_form_avg_contrib": a_key["key_players_form_avg_contrib"],
            "h_starting11_avg_minutes_7d": h_key["starting11_avg_minutes_7d"],
            "a_starting11_avg_minutes_7d": a_key["starting11_avg_minutes_7d"],
            "h_starting11_avg_matches_7d": h_key["starting11_avg_matches_7d"],
            "a_starting11_avg_matches_7d": a_key["starting11_avg_matches_7d"],
            # Differential features
            "diff_win_rate": home_features["win_rate"] - away_features["win_rate"],
            "diff_avg_gf": home_features["avg_gf"] - away_features["avg_gf"],
            "diff_avg_ga": home_features["avg_ga"] - away_features["avg_ga"],
            "diff_points_per_game": home_features["points_per_game"] - away_features["points_per_game"],
            "diff_second_half_gf": home_features["avg_second_half_gf"] - away_features["avg_second_half_gf"],
            "diff_days_since_last_match": h_sched["days_since_last_match"] - a_sched["days_since_last_match"],
            "diff_matches_last_7d": h_sched["matches_last_7d"] - a_sched["matches_last_7d"],
            "diff_key_players_absent": h_key["key_players_absent"] - a_key["key_players_absent"],
            # Labels
            "result": get_label_outcome(fx["ft_home"], fx["ft_away"]),
            "goals_home": fx["ft_home"],
            "goals_away": fx["ft_away"],
        }
        pretrain_rows.append(pretrain_row)
        
        # ===== In-play data: denser time checkpoints for sequence learning =====
        minutes = INPLAY_CHECKPOINT_MINUTES
        for min_checkpoint in minutes:
            inplay_features = InPlayFeatures.get_inplay_features(
                fx, min_checkpoint, fx["home_team_id"], fx["away_team_id"]
            )
        
            label = get_label_outcome(fx["ft_home"], fx["ft_away"])
            
            inplay_row = {
                "fixture_id": fx["fixture_id"],
                "date": fx["date"],
                "home_team": fx["home_team"],
                "away_team": fx["away_team"],
                # Pre-match features
                "h_win_rate": home_features["win_rate"],
                "h_avg_gf": home_features["avg_gf"],
                "h_avg_ga": home_features["avg_ga"],
                "a_win_rate": away_features["win_rate"],
                "a_avg_gf": away_features["avg_gf"],
                "a_avg_ga": away_features["avg_ga"],
                "h_points_per_game": home_features["points_per_game"],
                "a_points_per_game": away_features["points_per_game"],
                "h_clean_sheet_rate": home_features["clean_sheet_rate"],
                "a_clean_sheet_rate": away_features["clean_sheet_rate"],
                "h_avg_second_half_gf": home_features["avg_second_half_gf"],
                "a_avg_second_half_gf": away_features["avg_second_half_gf"],
                "h2h_home_win_rate": h2h_features["h2h_home_win_rate"],
                "h2h_draw_rate": h2h_features["h2h_draw_rate"],
                "h2h_home_goal_diff_avg": h2h_features["h2h_home_goal_diff_avg"],
                "h_days_since_last_match": h_sched["days_since_last_match"],
                "a_days_since_last_match": a_sched["days_since_last_match"],
                "h_matches_last_7d": h_sched["matches_last_7d"],
                "a_matches_last_7d": a_sched["matches_last_7d"],
                "season_progress": imp_feats["season_progress"],
                "h_rank": imp_feats["h_rank"],
                "a_rank": imp_feats["a_rank"],
                "importance_sum": imp_feats["importance_sum"],
                "importance_diff": imp_feats["importance_diff"],
                "h_key_players_started": h_key["key_players_started"],
                "a_key_players_started": a_key["key_players_started"],
                "h_key_players_absent": h_key["key_players_absent"],
                "a_key_players_absent": a_key["key_players_absent"],
                "h_starting11_avg_minutes_7d": h_key["starting11_avg_minutes_7d"],
                "a_starting11_avg_minutes_7d": a_key["starting11_avg_minutes_7d"],
                # In-play features
                **inplay_features,
                # Labels
                "result": label,
                "ft_home": fx["ft_home"],
                "ft_away": fx["ft_away"],
            }
            inplay_rows.append(inplay_row)
    
    pretrain_df = pd.DataFrame(pretrain_rows)
    inplay_df = pd.DataFrame(inplay_rows)
    
    return pretrain_df, inplay_df

def main():
    # Load data
    loader = FixtureLoader(DATA_DIR)
    fixtures = loader.load_all()
    
    if not fixtures:
        print("No fixtures found!")
        return
    
    # Build team history
    team_stats = TeamStats(fixtures)
    
    # Generate training datasets
    pretrain_df, inplay_df = generate_datasets(fixtures, team_stats)
    
    print(f"\nSaving pre-match dataset: {len(pretrain_df)} samples")
    pretrain_df.to_csv(OUT_DIR / "pretrain_dataset.csv", index=False, encoding="utf-8")
    
    print(f"Saving in-play dataset: {len(inplay_df)} samples")
    inplay_df.to_csv(OUT_DIR / "inplay_dataset.csv", index=False, encoding="utf-8")
    
    print("\n" + "="*60)
    print("✓ Feature extraction complete!")
    print(f"  - Pre-match: {len(pretrain_df)} samples")
    print(f"  - In-play: {len(inplay_df)} samples (6 time points per match)")
    print("="*60)


if __name__ == "__main__":
    main()
