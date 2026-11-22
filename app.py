import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson
from datetime import datetime

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

SEASON_OPTIONS = {
    "2026-27 (Actual)": 2027,
    "2025-26": 2026,
    "2024-25": 2025,
}

DISCRETE_METRICS = {"3PM", "STL", "BLK", "STOCKS", "TOV"}


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
                "abbr": team.get("abbreviation", ""),
                "logo": logos[0].get("href", "") if logos else "",
            }
        )
    return sorted(teams, key=lambda x: x["name"])


@st.cache_data(ttl=1800)
def fetch_today_scoreboard() -> list[dict]:
    """Obtiene los partidos programados para la jornada actual."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) == 2:
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            status = event.get("status", {}).get("type", {}).get("shortDetail", "")
            games.append(
                {
                    "label": f"{away['team']['abbreviation']} @ {home['team']['abbreviation']} ({status})",
                    "home_id": str(home["team"]["id"]),
                    "home_name": home["team"]["displayName"],
                    "away_id": str(away["team"]["id"]),
                    "away_name": away["team"]["displayName"],
                }
            )
    return games


@st.cache_data(ttl=3600)
def fetch_team_roster(team_id: str) -> list[dict]:
    """Obtiene la plantilla activa de un equipo específico."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

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


@st.cache_data(ttl=900)
def fetch_player_gamelog(athlete_id: str, season: int) -> pd.DataFrame:
    """Descarga y normaliza el registro juego por juego (Game Log) del jugador."""
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{athlete_id}/gamelog"
    resp = requests.get(url, params={"season": season}, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

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
        # (Grupo, Etiqueta del Logro, Columna, Umbral entero >= X)
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
        # El umbral continuo equivalente a >= target es (target - 0.5)
        line_equiv = target - 0.5
        res = calculate_prop_probabilities(df, col, line_equiv, weight_recent=weight_recent)
        prob = res["prob_over"]

        # Omitir logros totalmente fuera del perfil del jugador
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

    # Cálculo específico de Doble-Doble y Triple-Doble (combinando tasa empírica + probabilidad conjunta)
    prob_10_pts = calculate_prop_probabilities(df, "PTS", 9.5, weight_recent)["prob_over"] / 100.0
    prob_10_reb = calculate_prop_probabilities(df, "REB", 9.5, weight_recent)["prob_over"] / 100.0
    prob_10_ast = calculate_prop_probabilities(df, "AST", 9.5, weight_recent)["prob_over"] / 100.0

    # Probabilidad teórica de al menos 2 de (PTS, REB, AST) >= 10
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
st.title("🏀 NBA Player Analytics & Quantitative Props")
st.caption("Proyecciones estadísticas, análisis de logros (PTS, REB, AST, 3PM, DD/TD) y modelado de líneas.")

with st.sidebar:
    st.header("⚙️ Selección de Jugador")

    season_label = st.selectbox("Temporada NBA", list(SEASON_OPTIONS.keys()), index=0)
    selected_season = SEASON_OPTIONS[season_label]

    search_mode = st.radio(
        "Modo de búsqueda",
        ["Todos los Equipos (30)", "Partidos de Hoy (Cartelera)"],
        index=0,
    )

    try:
        teams_list = fetch_nba_teams()
    except Exception as exc:
        st.error(f"Error al cargar equipos de ESPN: {exc}")
        st.stop()

    selected_team_id = None
    selected_team_obj = None

    if search_mode == "Partidos de Hoy (Cartelera)":
        today_games = fetch_today_scoreboard()
        if today_games:
            game_labels = [g["label"] for g in today_games]
            chosen_game_label = st.selectbox("Partido de hoy", game_labels)
            game_obj = next(g for g in today_games if g["label"] == chosen_game_label)

            team_choice = st.radio(
                "Equipo del partido",
                [
                    (game_obj["away_id"], game_obj["away_name"]),
                    (game_obj["home_id"], game_obj["home_name"]),
                ],
                format_func=lambda x: x[1],
            )
            selected_team_id = team_choice[0]
            selected_team_obj = next((t for t in teams_list if t["id"] == selected_team_id), None)
        else:
            st.info("No hay partidos disponibles en la cartelera de hoy. Cambiando a lista general.")
            search_mode = "Todos los Equipos (30)"

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


# ---------------------------------------------------------
# CARGA DE DATOS DEL JUGADOR SELECCIONADO
# ---------------------------------------------------------
with st.spinner(f"Analizando estadísticas de {player_obj['name']}..."):
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
# NUEVA SECCIÓN: ¿QUÉ SE ESPERA DEL JUGADOR? (PROYECCIONES Y LOGROS)
# ---------------------------------------------------------
st.markdown("### 🔮 Análisis Predictivo: ¿Qué se espera del jugador?")

df_expectations = generate_player_expectations(df_filtered, weight_recent=recent_weight)
df_milestones = calculate_milestones_analysis(df_filtered, weight_recent=recent_weight)

# Veredicto automático con las principales expectativas y logros más sólidos
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
else:
    best_row = df_milestones.iloc[0] if not df_milestones.empty else None
    if best_row is not None:
        summary_lines.append(
            f"**Logro más probable:** **{best_row['Logro / Hito']}** "
            f"(Prob: {best_row['Probabilidad (%)']}% | L10: {best_row['Hit Rate L10 (%)']}%)"
        )

st.success("\n\n".join(summary_lines))

tab_proj, tab_milestones = st.tabs(
    ["📈 Proyecciones por Estadística (PTS, REB, AST, 3PM)", "🏆 Radar de Logros e Hitos (Dobles-Dobles y Escalones)"]
)

with tab_proj:
    st.caption(
        "El **Piso Seguro (75%)** indica el valor mínimo que el jugador supera en aproximadamente 3 de cada 4 partidos "
        "según su distribución estadística y forma reciente."
    )
    st.dataframe(
        df_expectations,
        use_container_width=True,
        hide_index=True,
    )

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
                "Probabilidad (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Hit Rate L10 (%)": st.column_config.ProgressColumn(
                "Hit Rate L10 (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

st.divider()

# ---------------------------------------------------------
# SIMULADOR CUANTITATIVO OVER / UNDER Y GRÁFICO DE TENDENCIA
# ---------------------------------------------------------
st.markdown("### 🎯 Simulador Personalizado de Líneas (Over / Under) y Tendencia")

sim_col1, sim_col2 = st.columns([1.2, 2.3])

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

    st.markdown(f"**Modelo aplicado:** `{sim_results['model_used']}`")
    st.markdown(f"**Proyección ponderada:** `{sim_results['projected_mean']}` (Desv. Est: `{sim_results['std_all']}`)")

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
        height=430,
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

st.dataframe(
    df_filtered[display_cols],
    use_container_width=True,
    hide_index=True,
)

csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar estadísticas en CSV",
    data=csv_bytes,
    file_name=f"{player_obj['name'].replace(' ', '_')}_NBA_{selected_season}.csv",
    mime="text/csv",
)
