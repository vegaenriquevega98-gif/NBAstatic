import streamlit as st
import requests
import pandas as pd
import numpy as np
import unicodedata
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, poisson
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="NBA Player Analytics Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

NBA_TZ = ZoneInfo("America/New_York")

SEASON_OPTIONS = {
    "2026-27 (Actual)": 2027,
    "2025-26": 2026,
    "2024-25": 2025,
}

DISCRETE_METRICS = {"3PM", "STL", "BLK", "STOCKS", "TOV"}

ODDS_MARKET_MAP = {
    "player_points": ("PTS", "Puntos (PTS)"),
    "player_rebounds": ("REB", "Rebotes (REB)"),
    "player_assists": ("AST", "Asistencias (AST)"),
    "player_threes": ("3PM", "Triples (3PM)"),
    "player_points_rebounds_assists": ("PRA", "Pts+Reb+Ast (PRA)"),
    "player_points_rebounds": ("PR", "Pts+Reb (PR)"),
    "player_points_assists": ("PA", "Pts+Ast (PA)"),
    "player_rebounds_assists": ("RA", "Reb+Ast (RA)"),
    "player_blocks_steals": ("STOCKS", "Robos+Tapones (STOCKS)"),
}


# ---------------------------------------------------------
# UTILIDADES DE NORMALIZACIÓN Y CÁLCULO DE VALOR (EV+)
# ---------------------------------------------------------
def _normalize_text(text: str) -> str:
    """Elimina tildes, puntos y sufijos (Jr., III) para empatar nombres entre APIs."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    for ch in [".", ",", "'", "-"]:
        ascii_str = ascii_str.replace(ch, " ")
    tokens = [t for t in ascii_str.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def evaluate_bet_value(model_prob_pct: float, decimal_odds: float) -> dict:
    """
    Calcula Probabilidad Implícita, Edge, Valor Esperado (EV%) y Stake Quarter-Kelly.
    """
    if decimal_odds <= 1.0 or model_prob_pct <= 0.0:
        return {
            "implied_prob": 0.0,
            "edge_pct": 0.0,
            "ev_pct": -100.0,
            "kelly_quarter_pct": 0.0,
            "is_value": False,
        }

    p = model_prob_pct / 100.0
    q = 1.0 - p
    b = decimal_odds - 1.0

    implied_prob = (1.0 / decimal_odds) * 100.0
    edge_pct = model_prob_pct - implied_prob
    ev_pct = ((p * decimal_odds) - 1.0) * 100.0

    kelly_full = max(0.0, ((b * p) - q) / b)
    kelly_quarter_pct = round(kelly_full * 0.25 * 100.0, 2)

    return {
        "implied_prob": round(implied_prob, 1),
        "edge_pct": round(edge_pct, 1),
        "ev_pct": round(ev_pct, 1),
        "kelly_quarter_pct": kelly_quarter_pct,
        "is_value": ev_pct > 0.0,
    }


def _parse_made_attempted(val: str) -> tuple[float, float]:
    """Convierte cadenas como '8-15' en (8.0, 15.0)."""
    try:
        made, att = str(val).split("-")
        return float(made), float(att)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------
# FUNCIONES DE EXTRACCIÓN DE DATOS (ESPN PUBLIC API)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_nba_teams() -> list[dict]:
    """Obtiene los 30 equipos de la NBA con sus IDs, abreviaturas y logos."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    teams = []
    for entry in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = entry.get("team", {})
        logos = team.get("logos", [])
        teams.append(
            {
                "id": str(team.get("id")),
                "name": team.get("displayName", "N/A"),
                "nickname": team.get("name", ""),
                "abbr": team.get("abbreviation", ""),
                "logo": logos[0].get("href", "") if logos else "",
            }
        )
    return sorted(teams, key=lambda x: x["name"])


@st.cache_data(ttl=180)
def fetch_scoreboard_by_date(date_yyyymmdd: str) -> list[dict]:
    """
    Obtiene los partidos programados para una fecha específica (formato YYYYMMDD)
    en horario oficial NBA (America/New_York).
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        resp = requests.get(
            url,
            params={"dates": date_yyyymmdd, "limit": 50},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    games = []
    for event in data.get("events", []):
        event_id = str(event.get("id", ""))
        comps = event.get("competitions", [])
        if not comps:
            continue
        competitors = comps[0].get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        home_abbr = home_team.get("abbreviation") or home_team.get("shortDisplayName", "LOC")
        away_abbr = away_team.get("abbreviation") or away_team.get("shortDisplayName", "VIS")

        status_type = event.get("status", {}).get("type", {})
        state = status_type.get("state", "pre")  # "pre", "in", "post"
        short_detail = status_type.get("shortDetail", "Programado")

        away_score = away.get("score", "")
        home_score = home.get("score", "")
        if state in {"in", "post"} and away_score != "" and home_score != "":
            score_tag = f"{away_abbr} {away_score} @ {home_abbr} {home_score}"
        else:
            score_tag = f"{away_abbr} @ {home_abbr}"

        games.append(
            {
                "event_id": event_id,
                "state": state,
                "status_detail": short_detail,
                "matchup": f"{away_abbr} @ {home_abbr}",
                "label": f"{score_tag}  •  {short_detail}",
                "home_id": str(home_team.get("id", "")),
                "home_name": home_team.get("displayName", home_abbr),
                "home_abbr": home_abbr,
                "home_logo": home_team.get("logo", ""),
                "away_id": str(away_team.get("id", "")),
                "away_name": away_team.get("displayName", away_abbr),
                "away_abbr": away_abbr,
                "away_logo": away_team.get("logo", ""),
            }
        )
    return games


def _fetch_roster_raw(team_id: str) -> list[dict]:
    """Extracción pura del roster de un equipo (apta para ejecución multihilo)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    raw_athletes = data.get("athletes", [])
    players = []

    for item in raw_athletes:
        group = item.get("items", [item]) if "items" in item else [item]
        for p in group:
            if not p.get("id"):
                continue
            headshot = p.get("headshot", {}).get("href", "")
            pos = p.get("position", {}).get("abbreviation", "-")
            players.append(
                {
                    "id": str(p["id"]),
                    "name": p.get("fullName") or p.get("displayName", "Jugador"),
                    "jersey": p.get("jersey", "-"),
                    "position": pos,
                    "age": p.get("age", "-"),
                    "height": p.get("displayHeight", "-"),
                    "weight": p.get("displayWeight", "-"),
                    "headshot": headshot,
                }
            )
    return sorted(players, key=lambda x: x["name"])


@st.cache_data(ttl=3600)
def fetch_team_roster(team_id: str) -> list[dict]:
    """Obtiene la plantilla activa de un equipo específico (con caché)."""
    return _fetch_roster_raw(team_id)


def _fetch_gamelog_raw(athlete_id: str, season: int) -> pd.DataFrame:
    """Extracción pura del Game Log de un jugador (apta para ejecución multihilo)."""
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{athlete_id}/gamelog"
    try:
        resp = requests.get(url, params={"season": season}, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return pd.DataFrame()

    labels = data.get("labels", [])
    events_meta = data.get("events", {})
    season_types = data.get("seasonTypes", [])

    rows = []
    for stype in season_types:
        phase_name = stype.get("displayName", "Regular Season")
        if "preseason" in phase_name.lower():
            continue

        for cat in stype.get("categories", []):
            for ev in cat.get("events", []):
                eid = str(ev.get("eventId", ""))
                stats_arr = ev.get("stats", [])
                if not labels or len(stats_arr) != len(labels):
                    continue

                stat_map = dict(zip(labels, stats_arr))
                meta = events_meta.get(eid, {})

                raw_date = meta.get("gameDate", "")
                try:
                    dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    date_str = raw_date[:10]

                opp = meta.get("opponent", {}).get("abbreviation", "N/A")
                at_vs = meta.get("atVs", "vs")
                result = meta.get("gameResult", "-")

                fgm, fga = _parse_made_attempted(stat_map.get("FG", "0-0"))
                tpm, tpa = _parse_made_attempted(stat_map.get("3PT", "0-0"))
                ftm, fta = _parse_made_attempted(stat_map.get("FT", "0-0"))

                pts = _safe_float(stat_map.get("PTS", 0))
                reb = _safe_float(stat_map.get("REB", 0))
                ast = _safe_float(stat_map.get("AST", 0))
                stl = _safe_float(stat_map.get("STL", 0))
                blk = _safe_float(stat_map.get("BLK", 0))
                tov = _safe_float(stat_map.get("TO", 0))
                mins = _safe_float(stat_map.get("MIN", 0))

                double_digit_cats = sum(1 for v in (pts, reb, ast, stl, blk) if v >= 10)

                rows.append(
                    {
                        "Fecha": date_str,
                        "Fase": phase_name,
                        "Rival": f"{at_vs} {opp}",
                        "Localía": "Local" if at_vs == "vs" else "Visitante",
                        "Resultado": result,
                        "MIN": mins,
                        "PTS": pts,
                        "REB": reb,
                        "AST": ast,
                        "3PM": tpm,
                        "3PA": tpa,
                        "FGM": fgm,
                        "FGA": fga,
                        "FG%": _safe_float(stat_map.get("FG%", 0)),
                        "3P%": _safe_float(stat_map.get("3P%", 0)),
                        "FTM": ftm,
                        "FTA": fta,
                        "FT%": _safe_float(stat_map.get("FT%", 0)),
                        "STL": stl,
                        "BLK": blk,
                        "TOV": tov,
                        "PRA": pts + reb + ast,
                        "PR": pts + reb,
                        "PA": pts + ast,
                        "RA": reb + ast,
                        "STOCKS": stl + blk,
                        "DD": 1 if double_digit_cats >= 2 else 0,
                        "TD": 1 if double_digit_cats >= 3 else 0,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Fecha", "Rival"]).sort_values("Fecha", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=900)
def fetch_player_gamelog(athlete_id: str, season: int) -> pd.DataFrame:
    """Descarga y normaliza el registro juego por juego (Game Log) del jugador."""
    return _fetch_gamelog_raw(athlete_id, season)


def _fetch_live_boxscore_raw(event_id: str) -> dict[str, dict]:
    """
    Extrae estadísticas en vivo o finales (PTS, REB, AST, 3PM, MIN, PRA)
    desde el Box Score oficial de ESPN para un partido iniciado/finalizado.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    try:
        resp = requests.get(url, params={"event": event_id}, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return {}

    live_map: dict[str, dict] = {}
    players_blocks = data.get("boxscore", {}).get("players", [])
    for team_block in players_blocks:
        stat_groups = team_block.get("statistics", [])
        if not stat_groups:
            continue
        labels = stat_groups[0].get("labels", [])
        athletes = stat_groups[0].get("athletes", [])
        if not labels:
            continue

        for ath_entry in athletes:
            ath = ath_entry.get("athlete", {})
            aid = str(ath.get("id", ""))
            stats_arr = ath_entry.get("stats", [])
            if not aid or not stats_arr or len(stats_arr) != len(labels):
                continue

            smap = dict(zip(labels, stats_arr))
            pts = _safe_float(smap.get("PTS", 0))
            reb = _safe_float(smap.get("REB", 0))
            ast = _safe_float(smap.get("AST", 0))
            tpm, _ = _parse_made_attempted(smap.get("3PT", "0-0"))
            mins = _safe_float(smap.get("MIN", 0))

            live_map[aid] = {
                "live_pra": round(pts + reb + ast, 1),
                "live_pts": int(pts),
                "live_reb": int(reb),
                "live_ast": int(ast),
                "live_3pm": int(tpm),
                "live_min": int(mins),
                "starter": bool(ath_entry.get("starter", False)),
            }
    return live_map


@st.cache_data(ttl=300)
def build_daily_pra_board(
    date_yyyymmdd: str,
    event_ids_tuple: tuple[str, ...],
    season: int,
    weight_recent: float = 0.40,
) -> pd.DataFrame:
    """
    Construye el tablero consolidado de jugadores y métricas PRA para los partidos
    de una fecha determinada, combinando datos en vivo/finales con proyecciones.
    """
    all_games = fetch_scoreboard_by_date(date_yyyymmdd)
    if not all_games:
        return pd.DataFrame()

    target_games = [g for g in all_games if not event_ids_tuple or g["event_id"] in event_ids_tuple]
    if not target_games:
        return pd.DataFrame()

    # 1. Extraer Box Scores en paralelo para partidos en curso o finalizados
    active_event_ids = [g["event_id"] for g in target_games if g["state"] in {"in", "post"}]
    live_stats_by_player: dict[str, dict] = {}
    if active_event_ids:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_fetch_live_boxscore_raw, active_event_ids))
            for res_dict in results:
                live_stats_by_player.update(res_dict)

    # 2. Mapear equipos involucrados en los partidos seleccionados
    team_tasks = []
    for g in target_games:
        state_badge = (
            f"🔴 EN VIVO ({g['status_detail']})"
            if g["state"] == "in"
            else (f"✅ FINAL ({g['status_detail']})" if g["state"] == "post" else f"🕒 {g['status_detail']}")
        )
        team_tasks.append(
            {
                "team_id": g["away_id"],
                "team_abbr": g["away_abbr"],
                "team_name": g["away_name"],
                "matchup": g["matchup"],
                "game_state": g["state"],
                "status_badge": state_badge,
            }
        )
        team_tasks.append(
            {
                "team_id": g["home_id"],
                "team_abbr": g["home_abbr"],
                "team_name": g["home_name"],
                "matchup": g["matchup"],
                "game_state": g["state"],
                "status_badge": state_badge,
            }
        )

    unique_team_ids = list({t["team_id"] for t in team_tasks if t["team_id"]})
    rosters_by_team: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        roster_results = list(ex.map(_fetch_roster_raw, unique_team_ids))
        for tid, plist in zip(unique_team_ids, roster_results):
            rosters_by_team[tid] = plist

    # 3. Preparar lista de jugadores a evaluar
    player_entries = []
    for tinfo in team_tasks:
        for p in rosters_by_team.get(tinfo["team_id"], []):
            player_entries.append({**p, **tinfo})

    if not player_entries:
        return pd.DataFrame()

    unique_player_ids = list({p["id"] for p in player_entries})
    gamelogs_by_player: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=14) as ex:
        gl_results = list(ex.map(lambda pid: _fetch_gamelog_raw(pid, season), unique_player_ids))
        for pid, df_gl in zip(unique_player_ids, gl_results):
            gamelogs_by_player[pid] = df_gl

    # 4. Consolidar métricas PRA por jugador
    rows = []
    for p in player_entries:
        pid = p["id"]
        df_gl = gamelogs_by_player.get(pid, pd.DataFrame())
        live_info = live_stats_by_player.get(pid)

        if not df_gl.empty:
            games_played = len(df_gl)
            min_avg = round(float(df_gl["MIN"].mean()), 1)
            pts_avg = round(float(df_gl["PTS"].mean()), 1)
            reb_avg = round(float(df_gl["REB"].mean()), 1)
            ast_avg = round(float(df_gl["AST"].mean()), 1)
            tpm_avg = round(float(df_gl["3PM"].mean()), 1)

            pra_series = df_gl["PRA"].astype(float)
            pra_all = float(pra_series.mean())
            pra_l10 = float(pra_series.head(10).mean())
            pra_l5 = float(pra_series.head(5).mean())
            pra_std = _get_effective_std(pra_series, pra_all)

            pra_proj = (1 - weight_recent) * pra_all + weight_recent * pra_l10
            pra_floor = max(0.0, float(norm.ppf(0.25, loc=pra_proj, scale=pra_std)))

            diff_l5 = pra_l5 - pra_all
            if pra_all > 0 and (pra_l5 >= pra_all * 1.12) and diff_l5 >= 1.0:
                trend = "🔥 En racha"
            elif pra_all > 0 and (pra_l5 <= pra_all * 0.88) and diff_l5 <= -1.0:
                trend = "❄️ A la baja"
            else:
                trend = "⚖️ Estable"
        else:
            games_played = 0
            min_avg = 0.0
            pts_avg = 0.0
            reb_avg = 0.0
            ast_avg = 0.0
            tpm_avg = 0.0
            pra_all = 0.0
            pra_l10 = 0.0
            pra_l5 = 0.0
            pra_proj = 0.0
            pra_floor = 0.0
            trend = "➖ Sin datos"

        live_pra_val = live_info["live_pra"] if live_info else np.nan
        live_detail = (
            f"{live_info['live_pts']}P / {live_info['live_reb']}R / {live_info['live_ast']}A ({live_info['live_min']}m)"
            if live_info
            else "-"
        )

        rows.append(
            {
                "Jugador": p["name"],
                "Pos": p["position"],
                "Equipo": p["team_abbr"],
                "Partido": p["matchup"],
                "Estado": p["status_badge"],
                "PJ": games_played,
                "MIN": min_avg,
                "PRA Partido (Vivo/Final)": live_pra_val,
                "Detalle Partido": live_detail,
                "PRA Proyectado": round(pra_proj, 1),
                "PRA Temp.": round(pra_all, 1),
                "PRA L10": round(pra_l10, 1),
                "PRA L5": round(pra_l5, 1),
                "Piso PRA (75%)": round(pra_floor, 1),
                "PTS": pts_avg,
                "REB": reb_avg,
                "AST": ast_avg,
                "3PM": tpm_avg,
                "Tendencia": trend,
            }
        )

    df_board = pd.DataFrame(rows)
    if df_board.empty:
        return df_board

    df_board = df_board.sort_values(
        by=["PRA Partido (Vivo/Final)", "PRA Proyectado", "PRA Temp."],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return df_board


# ---------------------------------------------------------
# EXTRACCIÓN DE CUOTAS DE PLAYER PROPS (THE ODDS API v4)
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_odds_api_events(api_key: str) -> tuple[list[dict], str | None]:
    """Lista los partidos próximos en The Odds API para localizar el event_id."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
        if resp.status_code == 401:
            return [], "API Key inválida o sin autorización en The Odds API."
        if resp.status_code == 429:
            return [], "Límite de peticiones alcanzado en The Odds API."
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        return resp.json(), f"OK (Peticiones restantes: {remaining})"
    except requests.RequestException as exc:
        return [], f"Error de conexión con The Odds API: {exc}"


@st.cache_data(ttl=1800)
def fetch_event_player_props(api_key: str, event_id: str, regions: str = "us,eu") -> tuple[dict, str]:
    """Descarga cuotas de Player Props de un partido específico."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    markets_str = ",".join(ODDS_MARKET_MAP.keys())
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets_str,
        "oddsFormat": "decimal",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return {}, f"Error HTTP {resp.status_code}: {resp.text[:160]}"
        remaining = resp.headers.get("x-requests-remaining", "?")
        return resp.json(), f"Peticiones restantes en el mes: {remaining}"
    except requests.RequestException as exc:
        return {}, f"Error de red: {exc}"


def parse_and_scan_player_odds(
    event_odds_json: dict,
    player_name: str,
    df_player: pd.DataFrame,
    weight_recent: float = 0.40,
) -> pd.DataFrame:
    """
    Cruza todas las cuotas extraídas de las casas de apuestas para el jugador
    contra las probabilidades del modelo estadístico y detecta errores (EV+).
    """
    target_norm = _normalize_text(player_name)
    bookmakers = event_odds_json.get("bookmakers", [])
    rows = []

    for book in bookmakers:
        book_title = book.get("title", book.get("key", "Casa"))
        for mkt in book.get("markets", []):
            mkt_key = mkt.get("key", "")
            if mkt_key not in ODDS_MARKET_MAP:
                continue
            metric_col, metric_label = ODDS_MARKET_MAP[mkt_key]
            if metric_col not in df_player.columns:
                continue

            for outcome in mkt.get("outcomes", []):
                desc_norm = _normalize_text(outcome.get("description", ""))
                if not desc_norm or (target_norm != desc_norm and target_norm not in desc_norm):
                    continue

                side = outcome.get("name", "")
                line = outcome.get("point")
                price = outcome.get("price")

                if line is None or price is None or side not in {"Over", "Under"}:
                    continue

                line_val = float(line)
                odds_val = float(price)

                sim = calculate_prop_probabilities(
                    df_player, metric_col, line_val, weight_recent=weight_recent
                )
                model_prob = sim["prob_over"] if side == "Over" else sim["prob_under"]
                fair_odds = sim["fair_odds_over"] if side == "Over" else sim["fair_odds_under"]
                hr_all = sim["hit_rate_all"] if side == "Over" else round(100.0 - sim["hit_rate_all"], 1)
                hr_l10 = sim["hit_rate_l10"] if side == "Over" else round(100.0 - sim["hit_rate_l10"], 1)

                val_metrics = evaluate_bet_value(model_prob, odds_val)

                if val_metrics["ev_pct"] >= 8.0 and val_metrics["edge_pct"] >= 4.0:
                    signal = "🔥 ERROR CLARO (EV+ Alto)"
                elif val_metrics["ev_pct"] > 0.0:
                    signal = "✅ VALOR POSITIVO (EV+)"
                else:
                    signal = "⚪ Sin Valor (EV-)"

                rows.append(
                    {
                        "Señal": signal,
                        "Casa": book_title,
                        "Mercado": metric_label,
                        "Selección": f"{side} {line_val}",
                        "Cuota Casa": round(odds_val, 2),
                        "Cuota Justa": fair_odds,
                        "Prob. Modelo (%)": model_prob,
                        "Prob. Casa (%)": val_metrics["implied_prob"],
                        "Edge (%)": val_metrics["edge_pct"],
                        "EV (%)": val_metrics["ev_pct"],
                        "Hit Rate Temp (%)": hr_all,
                        "Hit Rate L10 (%)": hr_l10,
                        "Stake Kelly 1/4 (%)": val_metrics["kelly_quarter_pct"],
                    }
                )

    if not rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(by=["EV (%)", "Edge (%)"], ascending=False).reset_index(drop=True)
    return df_res


# ---------------------------------------------------------
# CÁLCULOS ESTADÍSTICOS, PROYECCIONES Y LOGROS
# ---------------------------------------------------------
def _get_effective_std(series: pd.Series, mean_val: float) -> float:
    raw_std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    if pd.notna(raw_std) and raw_std > 0:
        return max(raw_std, 0.75)
    return max(mean_val * 0.25, 1.0)


def calculate_prop_probabilities(
    df: pd.DataFrame, metric: str, line: float, weight_recent: float = 0.40
) -> dict:
    """
    Calcula Hit Rates históricos y probabilidades estimadas (Normal y Poisson)
    combinando el rendimiento de toda la temporada con la forma reciente (últimos 10).
    """
    series_all = df[metric].astype(float)
    series_l10 = series_all.head(10)
    series_l5 = series_all.head(5)

    mean_all = float(series_all.mean())
    effective_std = _get_effective_std(series_all, mean_all)
    mean_l10 = float(series_l10.mean())
    mean_l5 = float(series_l5.mean())

    projected_mean = (1 - weight_recent) * mean_all + weight_recent * mean_l10

    prob_over_norm = float(1 - norm.cdf(line, loc=projected_mean, scale=effective_std))
    threshold_int = int(np.floor(line))
    prob_over_pois = float(1 - poisson.cdf(threshold_int, mu=max(projected_mean, 0.01)))

    prob_over = prob_over_pois if metric in DISCRETE_METRICS else prob_over_norm
    prob_over = float(np.clip(prob_over, 0.001, 0.999))
    prob_under = 1.0 - prob_over

    hit_rate_all = float((series_all > line).mean() * 100)
    hit_rate_l10 = float((series_l10 > line).mean() * 100)
    hit_rate_l5 = float((series_l5 > line).mean() * 100)

    fair_odds_over = round(1 / prob_over, 2) if prob_over > 0.01 else 99.0
    fair_odds_under = round(1 / prob_under, 2) if prob_under > 0.01 else 99.0

    return {
        "projected_mean": round(projected_mean, 2),
        "mean_all": round(mean_all, 2),
        "mean_l10": round(mean_l10, 2),
        "mean_l5": round(mean_l5, 2),
        "std_all": round(effective_std, 2),
        "hit_rate_all": round(hit_rate_all, 1),
        "hit_rate_l10": round(hit_rate_l10, 1),
        "hit_rate_l5": round(hit_rate_l5, 1),
        "prob_over": round(prob_over * 100, 1),
        "prob_under": round(prob_under * 100, 1),
        "fair_odds_over": fair_odds_over,
        "fair_odds_under": fair_odds_under,
        "model_used": "Poisson" if metric in DISCRETE_METRICS else "Normal Ponderado",
    }


def generate_player_expectations(df: pd.DataFrame, weight_recent: float = 0.40) -> pd.DataFrame:
    """
    Genera la tabla de proyección esperada por categoría (Puntos, Rebotes, Asistencias,
    Triples, Combinadas y Defensa) con piso seguro (75% prob.) y techo (25% prob.).
    """
    categories = [
        ("PTS", "Puntos (PTS)"),
        ("REB", "Rebotes (REB)"),
        ("AST", "Asistencias (AST)"),
        ("3PM", "Triples Anotados (3PM)"),
        ("PRA", "Puntos + Rebotes + Asist. (PRA)"),
        ("PR", "Puntos + Rebotes (PR)"),
        ("PA", "Puntos + Asistencias (PA)"),
        ("RA", "Rebotes + Asistencias (RA)"),
        ("STL", "Robos (STL)"),
        ("BLK", "Tapones (BLK)"),
        ("STOCKS", "Robos + Tapones (STOCKS)"),
    ]

    records = []
    for col, label in categories:
        s_all = df[col].astype(float)
        s_l10 = s_all.head(10)
        s_l5 = s_all.head(5)

        mean_all = float(s_all.mean())
        mean_l10 = float(s_l10.mean())
        mean_l5 = float(s_l5.mean())
        std_val = _get_effective_std(s_all, mean_all)

        proj = (1 - weight_recent) * mean_all + weight_recent * mean_l10

        if col in DISCRETE_METRICS:
            floor_val = float(poisson.ppf(0.25, mu=max(proj, 0.01)))
            ceil_val = float(poisson.ppf(0.75, mu=max(proj, 0.01)))
        else:
            floor_val = max(0.0, float(norm.ppf(0.25, loc=proj, scale=std_val)))
            ceil_val = max(floor_val, float(norm.ppf(0.75, loc=proj, scale=std_val)))

        diff_l5 = mean_l5 - mean_all
        if mean_all > 0 and (mean_l5 >= mean_all * 1.12) and diff_l5 >= 0.4:
            trend = "🔥 En racha"
        elif mean_all > 0 and (mean_l5 <= mean_all * 0.88) and diff_l5 <= -0.4:
            trend = "❄️ A la baja"
        else:
            trend = "⚖️ Estable"

        records.append(
            {
                "Categoría": label,
                "Prom. Temp": round(mean_all, 1),
                "Últ. 10": round(mean_l10, 1),
                "Últ. 5": round(mean_l5, 1),
                "Proyección Esperada": round(proj, 1),
                "Piso Seguro (75%)": round(floor_val, 1),
                "Techo Probable (25%)": round(ceil_val, 1),
                "Tendencia": trend,
            }
        )

    return pd.DataFrame(records)


def calculate_milestones_analysis(df: pd.DataFrame, weight_recent: float = 0.40) -> pd.DataFrame:
    """
    Evalúa hitos y logros concretos (15+/20+/25+/30+ PTS, 6+/8+/10+ REB, 4+/6+/8+/10+ AST,
    1+/2+/3+/4+ 3PM, Doble-Doble, Triple-Doble y combinadas).
    """
    milestone_specs = [
        ("Puntos", "10+ Puntos", "PTS", 10),
        ("Puntos", "15+ Puntos", "PTS", 15),
        ("Puntos", "20+ Puntos", "PTS", 20),
        ("Puntos", "25+ Puntos", "PTS", 25),
        ("Puntos", "30+ Puntos", "PTS", 30),
        ("Puntos", "35+ Puntos", "PTS", 35),
        ("Rebotes", "4+ Rebotes", "REB", 4),
        ("Rebotes", "6+ Rebotes", "REB", 6),
        ("Rebotes", "8+ Rebotes", "REB", 8),
        ("Rebotes", "10+ Rebotes", "REB", 10),
        ("Rebotes", "12+ Rebotes", "REB", 12),
        ("Asistencias", "4+ Asistencias", "AST", 4),
        ("Asistencias", "6+ Asistencias", "AST", 6),
        ("Asistencias", "8+ Asistencias", "AST", 8),
        ("Asistencias", "10+ Asistencias", "AST", 10),
        ("Triples", "1+ Triple Anotado (3PM)", "3PM", 1),
        ("Triples", "2+ Triples Anotados (3PM)", "3PM", 2),
        ("Triples", "3+ Triples Anotados (3PM)", "3PM", 3),
        ("Triples", "4+ Triples Anotados (3PM)", "3PM", 4),
        ("Triples", "5+ Triples Anotados (3PM)", "3PM", 5),
        ("Combinadas y Defensa", "20+ Pts+Reb+Ast (PRA)", "PRA", 20),
        ("Combinadas y Defensa", "25+ Pts+Reb+Ast (PRA)", "PRA", 25),
        ("Combinadas y Defensa", "30+ Pts+Reb+Ast (PRA)", "PRA", 30),
        ("Combinadas y Defensa", "40+ Pts+Reb+Ast (PRA)", "PRA", 40),
        ("Combinadas y Defensa", "1+ Robo (STL)", "STL", 1),
        ("Combinadas y Defensa", "1+ Tapón (BLK)", "BLK", 1),
        ("Combinadas y Defensa", "2+ Robos+Tapones (STOCKS)", "STOCKS", 2),
    ]

    rows = []
    for group, label, col, target in milestone_specs:
        line_equiv = target - 0.5
        res = calculate_prop_probabilities(df, col, line_equiv, weight_recent=weight_recent)
        prob = res["prob_over"]

        if prob < 4.0 and res["hit_rate_all"] == 0.0:
            continue

        if prob >= 75.0:
            verdict = "🟢 Muy Probable"
        elif prob >= 55.0:
            verdict = "🟡 Probable"
        elif prob >= 35.0:
            verdict = "🟠 Moderado"
        else:
            verdict = "🔴 Difícil"

        rows.append(
            {
                "Categoría": group,
                "Logro / Hito": label,
                "Probabilidad (%)": prob,
                "Hit Rate Temp (%)": res["hit_rate_all"],
                "Hit Rate L10 (%)": res["hit_rate_l10"],
                "Hit Rate L5 (%)": res["hit_rate_l5"],
                "Cuota Justa": res["fair_odds_over"],
                "Nivel": verdict,
            }
        )

    prob_10_pts = calculate_prop_probabilities(df, "PTS", 9.5, weight_recent)["prob_over"] / 100.0
    prob_10_reb = calculate_prop_probabilities(df, "REB", 9.5, weight_recent)["prob_over"] / 100.0
    prob_10_ast = calculate_prop_probabilities(df, "AST", 9.5, weight_recent)["prob_over"] / 100.0

    p_dd_theo = (
        prob_10_pts * prob_10_reb * (1 - prob_10_ast)
        + prob_10_pts * prob_10_ast * (1 - prob_10_reb)
        + prob_10_reb * prob_10_ast * (1 - prob_10_pts)
        + prob_10_pts * prob_10_reb * prob_10_ast
    )
    p_td_theo = prob_10_pts * prob_10_reb * prob_10_ast

    for special_col, special_name, p_theo in [
        ("DD", "Doble-Doble (2+ cats en 10+)", p_dd_theo),
        ("TD", "Triple-Doble (3+ cats en 10+)", p_td_theo),
    ]:
        hr_all = float(df[special_col].mean() * 100)
        hr_l10 = float(df[special_col].head(10).mean() * 100)
        hr_l5 = float(df[special_col].head(5).mean() * 100)
        emp_weighted = (1 - weight_recent) * (hr_all / 100.0) + weight_recent * (hr_l10 / 100.0)
        blended_prob = round(float(np.clip((0.5 * emp_weighted + 0.5 * p_theo) * 100, 0.1, 99.0)), 1)

        if blended_prob >= 75.0:
            verdict = "🟢 Muy Probable"
        elif blended_prob >= 55.0:
            verdict = "🟡 Probable"
        elif blended_prob >= 35.0:
            verdict = "🟠 Moderado"
        else:
            verdict = "🔴 Difícil"

        rows.append(
            {
                "Categoría": "Especiales (DD / TD)",
                "Logro / Hito": special_name,
                "Probabilidad (%)": blended_prob,
                "Hit Rate Temp (%)": round(hr_all, 1),
                "Hit Rate L10 (%)": round(hr_l10, 1),
                "Hit Rate L5 (%)": round(hr_l5, 1),
                "Cuota Justa": round(100.0 / blended_prob, 2) if blended_prob > 1.0 else 99.0,
                "Nivel": verdict,
            }
        )

    df_milestones = pd.DataFrame(rows)
    if not df_milestones.empty:
        df_milestones = df_milestones.sort_values(
            by=["Probabilidad (%)", "Hit Rate L10 (%)"], ascending=False
        ).reset_index(drop=True)
    return df_milestones


# ---------------------------------------------------------
# INTERFAZ DE USUARIO (SIDEBAR Y PANEL PRINCIPAL)
# ---------------------------------------------------------
st.title("🏀 NBA Player Analytics & EV+ Odds Scanner")
st.caption("Radar PRA por fecha, proyecciones estadísticas, logros (PTS, REB, AST, 3PM, DD/TD) y detección de errores de cuotas.")

today_nba_date = datetime.now(NBA_TZ).date()

with st.sidebar:
    st.header("⚙️ Selección de Jugador")

    season_label = st.selectbox("Temporada NBA", list(SEASON_OPTIONS.keys()), index=0)
    selected_season = SEASON_OPTIONS[season_label]

    search_mode = st.radio(
        "Modo de búsqueda",
        ["Partidos por Fecha (Cartelera)", "Todos los Equipos (30)"],
        index=0,
    )

    try:
        teams_list = fetch_nba_teams()
    except Exception as exc:
        st.error(f"Error al cargar equipos de ESPN: {exc}")
        st.stop()

    selected_team_id = None
    selected_team_obj = None

    if search_mode == "Partidos por Fecha (Cartelera)":
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            sidebar_date = st.date_input("Fecha de partidos", value=today_nba_date, key="sidebar_date")
        with col_d2:
            st.write("")
            if st.button("🔄", key="refresh_sb_games", help="Actualizar partidos desde ESPN"):
                fetch_scoreboard_by_date.clear()

        target_yyyymmdd = sidebar_date.strftime("%Y%m%d")
        slate_games = fetch_scoreboard_by_date(target_yyyymmdd)

        if slate_games:
            game_labels = [g["label"] for g in slate_games]
            chosen_game_label = st.selectbox("Partido", game_labels)
            game_obj = next(g for g in slate_games if g["label"] == chosen_game_label)

            team_choice = st.radio(
                "Equipo del partido",
                [
                    (game_obj["away_id"], f"{game_obj['away_name']} (Visitante)"),
                    (game_obj["home_id"], f"{game_obj['home_name']} (Local)"),
                ],
                format_func=lambda x: x[1],
            )
            selected_team_id = team_choice[0]
            selected_team_obj = next((t for t in teams_list if t["id"] == selected_team_id), None)

            if not selected_team_obj:
                is_home = selected_team_id == game_obj["home_id"]
                selected_team_obj = {
                    "id": selected_team_id,
                    "name": game_obj["home_name"] if is_home else game_obj["away_name"],
                    "nickname": game_obj["home_abbr"] if is_home else game_obj["away_abbr"],
                    "abbr": game_obj["home_abbr"] if is_home else game_obj["away_abbr"],
                    "logo": game_obj["home_logo"] if is_home else game_obj["away_logo"],
                }
        else:
            st.warning(
                f"No hay partidos programados en ESPN para **{sidebar_date.strftime('%Y-%m-%d')}**. "
                "Elige otra fecha arriba o cambia a **Todos los Equipos (30)**."
            )
            st.stop()

    if search_mode == "Todos los Equipos (30)":
        team_names = [t["name"] for t in teams_list]
        default_idx = team_names.index("Los Angeles Lakers") if "Los Angeles Lakers" in team_names else 0
        chosen_team_name = st.selectbox("Franquicia NBA", team_names, index=default_idx)
        selected_team_obj = next(t for t in teams_list if t["name"] == chosen_team_name)
        selected_team_id = selected_team_obj["id"]

    roster = fetch_team_roster(selected_team_id)
    if not roster:
        st.warning("No se encontró plantilla activa para este equipo.")
        st.stop()

    player_names = [f"{p['name']} (#{p['jersey']} - {p['position']})" for p in roster]
    chosen_player_label = st.selectbox("Jugador", player_names)
    player_obj = roster[player_names.index(chosen_player_label)]

    st.divider()
    st.subheader("🎯 Parámetros del Modelo")
    sample_window = st.slider("Ventana de partidos recientes (gráfico)", min_value=5, max_value=40, value=15)
    recent_weight = (
        st.slider(
            "Peso de los últimos 10 juegos en la proyección (%)",
            min_value=0,
            max_value=80,
            value=40,
            step=5,
        )
        / 100.0
    )
    location_filter = st.selectbox("Filtro de Localía", ["Todos", "Local", "Visitante"])

    st.divider()
    st.subheader("🔑 API de Cuotas (The Odds API)")
    secret_odds_key = st.secrets.get("ODDS_API_KEY", "") if hasattr(st, "secrets") else ""
    manual_odds_key = st.text_input(
        "ODDS_API_KEY (opcional si está en Secrets)",
        value=secret_odds_key,
        type="password",
        help="Obtén una clave gratuita en the-odds-api.com (500 peticiones/mes).",
    )
    active_odds_key = manual_odds_key.strip()


# ---------------------------------------------------------
# APARTADO: RADAR PRA DE LA JORNADA (JUGADORES Y PARTIDOS POR FECHA)
# ---------------------------------------------------------
st.markdown("## 🗓️ Radar PRA de la Jornada: Jugadores y Partidos por Fecha")
st.caption(
    "Selecciona una fecha para consultar los partidos del día y el rendimiento **PRA (Puntos + Rebotes + Asistencias)** "
    "en vivo, finalizado y proyectado de todos los jugadores participantes."
)

pra_ctrl1, pra_ctrl2, pra_ctrl3, pra_ctrl4 = st.columns([1.3, 2.2, 1.5, 0.8])

with pra_ctrl1:
    pra_board_date = st.date_input(
        "Seleccionar fecha",
        value=today_nba_date,
        key="pra_board_date_picker",
    )

pra_date_str = pra_board_date.strftime("%Y%m%d")
daily_games = fetch_scoreboard_by_date(pra_date_str)

with pra_ctrl2:
    if daily_games:
        game_filter_options = ["Todos los partidos del día"] + [g["label"] for g in daily_games]
        chosen_pra_game = st.selectbox(
            f"Partidos del {pra_board_date.strftime('%d/%m/%Y')} ({len(daily_games)})",
            game_filter_options,
            key="pra_board_game_filter",
        )
    else:
        chosen_pra_game = "Todos los partidos del día"
        st.selectbox(
            f"Partidos del {pra_board_date.strftime('%d/%m/%Y')} (0)",
            ["Sin partidos programados"],
            disabled=True,
        )

with pra_ctrl3:
    only_active_rotation = st.checkbox(
        "Solo jugadores activos en temporada",
        value=True,
        help="Oculta jugadores sin minutos jugados aún en la temporada seleccionada.",
    )

with pra_ctrl4:
    st.write("")
    if st.button("🔄 Actualizar", key="refresh_pra_board", help="Refrescar marcadores y estadísticas PRA"):
        fetch_scoreboard_by_date.clear()
        build_daily_pra_board.clear()

if not daily_games:
    st.info(
        f"📅 No hay partidos de la NBA programados para el **{pra_board_date.strftime('%Y-%m-%d')}**. "
        "Selecciona otra fecha en el calendario de arriba para explorar la cartelera."
    )
else:
    if chosen_pra_game == "Todos los partidos del día":
        selected_event_ids = tuple(g["event_id"] for g in daily_games)
    else:
        matched_g = next((g for g in daily_games if g["label"] == chosen_pra_game), None)
        selected_event_ids = (matched_g["event_id"],) if matched_g else tuple()

    with st.spinner(f"Cargando jugadores y métricas PRA del {pra_board_date.strftime('%Y-%m-%d')}..."):
        df_pra_board = build_daily_pra_board(
            pra_date_str,
            selected_event_ids,
            selected_season,
            weight_recent=recent_weight,
        )

    if df_pra_board.empty:
        st.warning("No se encontraron registros de jugadores para los partidos seleccionados.")
    else:
        df_pra_view = df_pra_board.copy()
        if only_active_rotation:
            df_pra_view = df_pra_view[
                (df_pra_view["PJ"] > 0) | (df_pra_view["PRA Partido (Vivo/Final)"].notna())
            ].reset_index(drop=True)

        # Métricas resumen de la jornada
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Partidos en Cartelera", f"{len(daily_games)}")
        m_col2.metric("Jugadores Listados", f"{len(df_pra_view)}")

        if not df_pra_view.empty:
            top_proj_row = df_pra_view.sort_values("PRA Proyectado", ascending=False).iloc[0]
            m_col3.metric(
                "Mayor PRA Proyectado",
                f"{top_proj_row['PRA Proyectado']} PRA",
                f"{top_proj_row['Jugador']} ({top_proj_row['Equipo']})",
            )

            live_subset = df_pra_view[df_pra_view["PRA Partido (Vivo/Final)"].notna()]
            if not live_subset.empty:
                top_live_row = live_subset.sort_values("PRA Partido (Vivo/Final)", ascending=False).iloc[0]
                m_col4.metric(
                    "Líder PRA en Vivo / Final",
                    f"{top_live_row['PRA Partido (Vivo/Final)']} PRA",
                    f"{top_live_row['Jugador']} ({top_live_row['Equipo']})",
                )
            else:
                m_col4.metric("Estado de Jornada", "Pre-partido", "Sin juegos iniciados aún")

        st.dataframe(
            df_pra_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "PRA Partido (Vivo/Final)": st.column_config.NumberColumn(
                    "PRA Partido (Vivo/Final)",
                    format="%.1f",
                    help="Suma real de Puntos + Rebotes + Asistencias en el partido del día (si ya inició o finalizó).",
                ),
                "PRA Proyectado": st.column_config.NumberColumn(
                    "PRA Proyectado",
                    format="%.1f",
                ),
                "PRA Temp.": st.column_config.NumberColumn(
                    "PRA Temp.",
                    format="%.1f",
                ),
                "PRA L10": st.column_config.NumberColumn(
                    "PRA L10",
                    format="%.1f",
                ),
                "PRA L5": st.column_config.NumberColumn(
                    "PRA L5",
                    format="%.1f",
                ),
            },
        )

        pra_csv = df_pra_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Descargar Tabla PRA del {pra_board_date.strftime('%Y-%m-%d')} (CSV)",
            data=pra_csv,
            file_name=f"NBA_PRA_Jornada_{pra_date_str}.csv",
            mime="text/csv",
            key="download_pra_board_csv",
        )

st.divider()

# ---------------------------------------------------------
# CARGA DE DATOS DEL JUGADOR SELECCIONADO (ANÁLISIS INDIVIDUAL)
# ---------------------------------------------------------
with st.spinner(f"Analizando estadísticas individuales de {player_obj['name']}..."):
    df_games = fetch_player_gamelog(player_obj["id"], selected_season)

if df_games.empty:
    st.warning(
        f"No hay registros de partidos disponibles para **{player_obj['name']}** "
        f"en la temporada **{season_label}**. Prueba seleccionando una temporada anterior en el panel lateral."
    )
    st.stop()

if location_filter != "Todos":
    df_filtered = df_games[df_games["Localía"] == location_filter].reset_index(drop=True)
else:
    df_filtered = df_games.copy()

if df_filtered.empty:
    st.warning(f"No hay partidos registrados bajo el filtro '{location_filter}'.")
    st.stop()

# ---------------------------------------------------------
# CABECERA DEL PERFIL DEL JUGADOR
# ---------------------------------------------------------
col_img, col_bio, col_team = st.columns([1.2, 4.8, 1.5])

with col_img:
    if player_obj["headshot"]:
        st.image(player_obj["headshot"], width=120)
    else:
        st.markdown("### 🏀")

with col_bio:
    st.subheader(f"{player_obj['name']}  |  #{player_obj['jersey']}")
    st.markdown(
        f"**Equipo:** {selected_team_obj['name'] if selected_team_obj else ''}  •  "
        f"**Posición:** {player_obj['position']}  •  "
        f"**Edad:** {player_obj['age']}  •  "
        f"**Estatura/Peso:** {player_obj['height']} / {player_obj['weight']}"
    )
    st.markdown(f"**Partidos analizados ({location_filter}):** `{len(df_filtered)}` juegos en {season_label}")

with col_team:
    if selected_team_obj and selected_team_obj.get("logo"):
        st.image(selected_team_obj["logo"], width=85)

st.divider()

# ---------------------------------------------------------
# TARJETAS DE PROMEDIOS (TEMPORADA VS ÚLTIMOS 5 PARTIDOS)
# ---------------------------------------------------------
st.markdown("### 📊 Promedios Generales vs. Forma Reciente (Últimos 5 Juegos)")

df_l5 = df_filtered.head(5)
kpi_cols = st.columns(7)

kpi_specs = [
    ("PTS", "Puntos"),
    ("REB", "Rebotes"),
    ("AST", "Asistencias"),
    ("PRA", "Pts+Reb+Ast"),
    ("3PM", "Triples"),
    ("STOCKS", "Rob+Tap"),
    ("MIN", "Minutos"),
]

for idx, (col_key, label) in enumerate(kpi_specs):
    season_avg = df_filtered[col_key].mean()
    l5_avg = df_l5[col_key].mean()
    delta_val = l5_avg - season_avg
    kpi_cols[idx].metric(
        label=label,
        value=f"{season_avg:.1f}",
        delta=f"{delta_val:+.1f} (L5)",
    )

eff_cols = st.columns(5)
eff_cols[0].metric("FG% (Campo)", f"{df_filtered['FG%'].mean():.1f}%")
eff_cols[1].metric("3P% (Triples)", f"{df_filtered['3P%'].mean():.1f}%")
eff_cols[2].metric("FT% (Libres)", f"{df_filtered['FT%'].mean():.1f}%")
eff_cols[3].metric("Doble-Doble %", f"{df_filtered['DD'].mean() * 100:.1f}%")
eff_cols[4].metric("Triple-Doble %", f"{df_filtered['TD'].mean() * 100:.1f}%")

st.divider()

# ---------------------------------------------------------
# ANÁLISIS PREDICTIVO: ¿QUÉ SE ESPERA DEL JUGADOR?
# ---------------------------------------------------------
st.markdown("### 🔮 Análisis Predictivo: ¿Qué se espera del jugador?")

df_expectations = generate_player_expectations(df_filtered, weight_recent=recent_weight)
df_milestones = calculate_milestones_analysis(df_filtered, weight_recent=recent_weight)

exp_map = dict(zip(df_expectations["Categoría"], df_expectations["Proyección Esperada"]))
floor_map = dict(zip(df_expectations["Categoría"], df_expectations["Piso Seguro (75%)"]))

top_picks = df_milestones[
    (df_milestones["Probabilidad (%)"] >= 65.0) & (df_milestones["Hit Rate L10 (%)"] >= 60.0)
].head(4)

summary_lines = [
    f"**Línea Estadística Esperada para {player_obj['name']}:** "
    f"**{exp_map['Puntos (PTS)']} PTS** (piso {floor_map['Puntos (PTS)']})  |  "
    f"**{exp_map['Rebotes (REB)']} REB** (piso {floor_map['Rebotes (REB)']})  |  "
    f"**{exp_map['Asistencias (AST)']} AST** (piso {floor_map['Asistencias (AST)']})  |  "
    f"**{exp_map['Triples Anotados (3PM)']} Triples**  |  "
    f"**{exp_map['Puntos + Rebotes + Asist. (PRA)']} PRA**"
]

if not top_picks.empty:
    picks_str = "  •  ".join(
        [
            f"**{r['Logro / Hito']}** (Prob: {r['Probabilidad (%)']}% | L10: {r['Hit Rate L10 (%)']}%)"
            for _, r in top_picks.iterrows()
        ]
    )
    summary_lines.append(f"**Logros con mayor respaldo estadístico:** {picks_str}")

st.success("\n\n".join(summary_lines))

tab_proj, tab_milestones = st.tabs(
    ["📈 Proyecciones por Estadística (PTS, REB, AST, 3PM)", "🏆 Radar de Logros e Hitos (Dobles-Dobles y Escalones)"]
)

with tab_proj:
    st.caption(
        "El **Piso Seguro (75%)** indica el valor mínimo que el jugador supera en aproximadamente 3 de cada 4 partidos "
        "según su distribución estadística y forma reciente."
    )
    st.dataframe(df_expectations, use_container_width=True, hide_index=True)

with tab_milestones:
    filter_cat = st.selectbox(
        "Filtrar logros por categoría",
        ["Todos los Logros", "Puntos", "Rebotes", "Asistencias", "Triples", "Combinadas y Defensa", "Especiales (DD / TD)"],
        index=0,
    )

    df_m_view = df_milestones.copy()
    if filter_cat != "Todos los Logros":
        df_m_view = df_m_view[df_m_view["Categoría"] == filter_cat]

    st.dataframe(
        df_m_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probabilidad (%)": st.column_config.ProgressColumn(
                "Probabilidad (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Hit Rate L10 (%)": st.column_config.ProgressColumn(
                "Hit Rate L10 (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

st.divider()

# ---------------------------------------------------------
# ESCÁNER DE CUOTAS Y DETECCIÓN DE ERRORES (EV+)
# ---------------------------------------------------------
st.markdown("### 💰 Escáner de Cuotas y Detección de Errores (Value Bets / EV+)")
st.caption(
    "Compara las cuotas ofrecidas por las casas de apuestas contra la probabilidad estadística del modelo. "
    "Un **EV% positivo** y un **Edge% > 3%** indican una cuota mal calibrada a favor del apostador."
)

if not active_odds_key:
    st.info(
        "🔑 **Modo sin API Key activa:** Puedes ingresar cuotas manualmente en el **Simulador de Líneas** de abajo "
        "para calcular el **EV%** y el **Stake Kelly**, o configurar `ODDS_API_KEY` en los Secrets de Streamlit "
        "para escanear automáticamente DraftKings, FanDuel, BetMGM y otras casas."
    )
else:
    col_opt1, col_opt2, col_opt3 = st.columns([1.5, 1.5, 2.0])
    with col_opt1:
        regions_sel = st.selectbox(
            "Región de Casas de Apuestas",
            ["us,eu", "us", "eu", "uk"],
            index=0,
            help="'us,eu' incluye casas americanas (DraftKings, FanDuel, BetMGM) y europeas.",
        )
    with col_opt2:
        only_ev_plus = st.checkbox("Mostrar solo errores con valor (EV+ > 0%)", value=False)
    with col_opt3:
        st.write("")
        scan_clicked = st.button("🔍 Escanear Cuotas de Hoy y Buscar Errores (EV+)", type="primary")

    if scan_clicked:
        with st.spinner("Consultando líneas y cuotas de Player Props en tiempo real..."):
            events_list, status_msg = fetch_odds_api_events(active_odds_key)
            if not events_list:
                st.error(status_msg or "No se encontraron partidos activos en The Odds API.")
            else:
                team_nick = _normalize_text(selected_team_obj.get("nickname", "")) if selected_team_obj else ""
                matched_event = None
                for ev in events_list:
                    h_norm = _normalize_text(ev.get("home_team", ""))
                    a_norm = _normalize_text(ev.get("away_team", ""))
                    if team_nick and (team_nick in h_norm or team_nick in a_norm):
                        matched_event = ev
                        break

                if not matched_event:
                    st.warning(
                        f"El equipo **{selected_team_obj['name']}** no tiene un partido abierto con mercados "
                        "de Player Props publicados en este momento (las casas suelen abrirlos horas antes del salto inicial)."
                    )
                else:
                    event_data, quota_msg = fetch_event_player_props(
                        active_odds_key, matched_event["id"], regions=regions_sel
                    )
                    st.caption(f"Partido detectado: **{matched_event['away_team']} @ {matched_event['home_team']}**  •  {quota_msg}")

                    df_odds_scan = parse_and_scan_player_odds(
                        event_data, player_obj["name"], df_filtered, weight_recent=recent_weight
                    )

                    if df_odds_scan.empty:
                        st.warning(
                            f"Aún no hay cuotas publicadas específicamente para **{player_obj['name']}** "
                            "en este partido (puede estar marcado como cuestionable/lesionado o las líneas aún no abren)."
                        )
                    else:
                        if only_ev_plus:
                            df_odds_scan = df_odds_scan[df_odds_scan["EV (%)"] > 0.0].reset_index(drop=True)

                        ev_count = int((df_odds_scan["EV (%)"] > 0.0).sum())
                        if ev_count > 0:
                            best_bet = df_odds_scan.iloc[0]
                            st.success(
                                f"🚨 **Se detectaron {ev_count} cuotas con Valor Esperado Positivo (EV+)** para {player_obj['name']}. "
                                f"Mejor oportunidad: **{best_bet['Mercado']} — {best_bet['Selección']}** en **{best_bet['Casa']}** "
                                f"a cuota **{best_bet['Cuota Casa']}** (Cuota justa: {best_bet['Cuota Justa']} | "
                                f"**EV: +{best_bet['EV (%)']}%** | Edge: +{best_bet['Edge (%)']}%)."
                            )
                        else:
                            st.info("No se detectaron cuotas con EV+ bajo los parámetros actuales.")

                        st.dataframe(df_odds_scan, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# SIMULADOR CUANTITATIVO OVER / UNDER + EVALUADOR MANUAL EV+
# ---------------------------------------------------------
st.markdown("### 🎯 Simulador Personalizado de Líneas (Over / Under) y Evaluador Manual de Cuotas")

sim_col1, sim_col2 = st.columns([1.25, 2.25])

available_metrics = ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "3PM", "STOCKS", "STL", "BLK"]

with sim_col1:
    selected_metric = st.selectbox("Estadística a evaluar", available_metrics, index=0)
    default_line = round(float(df_filtered[selected_metric].mean()) * 2) / 2
    target_line = st.number_input(
        f"Línea objetivo ({selected_metric})",
        min_value=0.0,
        max_value=100.0,
        value=max(0.5, default_line - 0.5),
        step=0.5,
    )

    sim_results = calculate_prop_probabilities(
        df_filtered, selected_metric, target_line, weight_recent=recent_weight
    )

    st.markdown(f"**Modelo aplicado:** `{sim_results['model_used']}`  •  **Proyección:** `{sim_results['projected_mean']}`")

    p_col1, p_col2 = st.columns(2)
    p_col1.metric(
        f"Prob. OVER {target_line}",
        f"{sim_results['prob_over']}%",
        f"Cuota justa: {sim_results['fair_odds_over']}",
    )
    p_col2.metric(
        f"Prob. UNDER {target_line}",
        f"{sim_results['prob_under']}%",
        f"Cuota justa: {sim_results['fair_odds_under']}",
    )

    st.markdown("**Comprobar cuota de tu casa de apuestas:**")
    odds_in1, odds_in2 = st.columns(2)
    book_odds_over = odds_in1.number_input("Cuota OVER (Decimal)", min_value=1.01, max_value=50.0, value=1.87, step=0.02)
    book_odds_under = odds_in2.number_input("Cuota UNDER (Decimal)", min_value=1.01, max_value=50.0, value=1.87, step=0.02)

    ev_over = evaluate_bet_value(sim_results["prob_over"], book_odds_over)
    ev_under = evaluate_bet_value(sim_results["prob_under"], book_odds_under)
    vig_pct = round(((1 / book_odds_over) + (1 / book_odds_under) - 1) * 100, 2)

    if ev_over["is_value"]:
        st.success(
            f"✅ **VALOR EN OVER {target_line}:** EV **+{ev_over['ev_pct']}%** | "
            f"Edge **+{ev_over['edge_pct']}%** | Stake Kelly 1/4: **{ev_over['kelly_quarter_pct']}%**"
        )
    elif ev_under["is_value"]:
        st.success(
            f"✅ **VALOR EN UNDER {target_line}:** EV **+{ev_under['ev_pct']}%** | "
            f"Edge **+{ev_under['edge_pct']}%** | Stake Kelly 1/4: **{ev_under['kelly_quarter_pct']}%**"
        )
    else:
        st.caption(
            f"Sin valor matemático (EV Over: {ev_over['ev_pct']}% | EV Under: {ev_under['ev_pct']}% | Comisión/Vig de la casa: {vig_pct}%)."
        )

    st.info(
        f"**Hit Rate Histórico (Superó {target_line}):**\n"
        f"- Temporada: **{sim_results['hit_rate_all']}%** ({int((df_filtered[selected_metric] > target_line).sum())}/{len(df_filtered)})\n"
        f"- Últimos 10: **{sim_results['hit_rate_l10']}%**\n"
        f"- Últimos 5: **{sim_results['hit_rate_l5']}%**"
    )

with sim_col2:
    plot_df = df_filtered.head(sample_window).iloc[::-1].copy()
    plot_df["Etiqueta"] = plot_df["Fecha"].str[5:] + " " + plot_df["Rival"]

    bar_colors = [
        "#00C853" if val > target_line else "#FF5252"
        for val in plot_df[selected_metric]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["Etiqueta"],
            y=plot_df[selected_metric],
            marker_color=bar_colors,
            text=plot_df[selected_metric],
            textposition="outside",
            name=selected_metric,
        )
    )

    fig.add_hline(
        y=target_line,
        line_dash="dash",
        line_color="#FFD600",
        line_width=2.5,
        annotation_text=f"Línea: {target_line}",
        annotation_position="top left",
    )

    fig.add_hline(
        y=sim_results["mean_all"],
        line_dash="dot",
        line_color="#40C4FF",
        line_width=2,
        annotation_text=f"Prom. Temp: {sim_results['mean_all']}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Últimos {len(plot_df)} partidos — {selected_metric} vs. Línea {target_line}",
        xaxis_title="Partido",
        yaxis_title=selected_metric,
        template="plotly_dark",
        height=470,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# TABLA DETALLADA DE GAME LOG Y EXPORTACIÓN CSV
# ---------------------------------------------------------
st.markdown("### 📋 Registro Completo de Partidos (Game Log)")

display_cols = [
    "Fecha", "Rival", "Localía", "Resultado", "MIN",
    "PTS", "REB", "AST", "PRA", "3PM", "3PA", "3P%",
    "FGM", "FGA", "FG%", "STL", "BLK", "TOV"
]

st.dataframe(df_filtered[display_cols], use_container_width=True, hide_index=True)

csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar estadísticas en CSV",
    data=csv_bytes,
    file_name=f"{player_obj['name'].replace(' ', '_')}_NBA_{selected_season}.csv",
    mime="text/csv",
)
