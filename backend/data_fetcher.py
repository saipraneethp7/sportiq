import time
import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players

# Team abbreviation to full name mapping
ABBREV_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

NAME_TO_ABBREV = {v: k for k, v in ABBREV_TO_NAME.items()}

def find_player(name: str) -> dict:
    """Search for a player by name and return their basic info."""
    results = players.find_players_by_full_name(name)
    if not results:
        return None
    return results[0]

def get_player_game_logs(player_id: int, season: str = '2024-25') -> pd.DataFrame:
    """Fetch a player's game by game logs for a season."""
    time.sleep(0.6)
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    )
    games = gamelog.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE').reset_index(drop=True)
    return games

def get_team_defense_ratings(season: str = '2024-25') -> dict:
    """Fetch defensive ratings for all 30 NBA teams."""
    time.sleep(0.6)
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense='Defense'
    )
    df = team_stats.get_data_frames()[0]
    return dict(zip(df['TEAM_NAME'], df['DEF_RATING']))

def calculate_features(games: pd.DataFrame, defense_ratings: dict) -> pd.DataFrame:
    """
    Take raw game logs and calculate all 5 ML features.
    This is the same feature engineering we did in the notebook,
    now packaged as a reusable function.
    """
    # Feature 1: Home or Away
    games['IS_HOME'] = games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # Feature 2: Days of rest
    games['DAYS_REST'] = games['GAME_DATE'].diff().dt.days.fillna(3)

    # Feature 3: Opponent defensive rating
    games['OPPONENT'] = games['MATCHUP'].apply(lambda x: x.split()[-1])
    games['OPPONENT_NAME'] = games['OPPONENT'].map(ABBREV_TO_NAME)
    games['OPP_DEF_RATING'] = games['OPPONENT_NAME'].map(defense_ratings)

    # Feature 4: Recent form
    games['FORM_5'] = games['PTS'].rolling(5).mean().shift(1)

    # Feature 5: Team momentum
    games['WIN'] = games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    games['WIN_RATE_5'] = games['WIN'].rolling(5).mean().shift(1)

    return games

def get_current_features(player_id: int, season: str = '2024-25') -> dict:
    """
    Get the CURRENT feature values for a player.
    These are the features we feed into the model for prediction.
    """
    games = get_player_game_logs(player_id, season)
    defense_ratings = get_team_defense_ratings(season)
    games = calculate_features(games, defense_ratings)

    # Get the most recent game's calculated features
    # This tells us the player's current form and team momentum
    latest = games.dropna(subset=['FORM_5', 'WIN_RATE_5']).iloc[-1]

    return {
        "recent_form": round(float(latest['FORM_5']), 1),
        "win_rate_5": round(float(latest['WIN_RATE_5']), 2),
        "days_rest": int(latest['DAYS_REST']),
        "games_played": len(games),
        "season_avg_pts": round(games['PTS'].mean(), 1)
    }

def get_player_stats_summary(player_id: int, season: str = '2024-25') -> dict:
    """
    Get a clean summary of a player's season stats.
    This powers the player dashboard in our frontend.
    """
    games = get_player_game_logs(player_id, season)

    # Calculate per game averages
    avg_pts = games['PTS'].mean().round(1)
    avg_ast = games['AST'].mean().round(1)
    avg_reb = games['REB'].mean().round(1)
    avg_fg = (games['FG_PCT'].mean() * 100).round(1)
    avg_fg3 = (games['FG3_PCT'].mean() * 100).round(1)

    # Win loss record
    wins = (games['WL'] == 'W').sum()
    losses = (games['WL'] == 'L').sum()

    # Last 5 games
    last5 = games.tail(5)[['GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'REB', 'WL']].copy()
    last5['GAME_DATE'] = last5['GAME_DATE'].dt.strftime('%b %d')

    return {
        "games_played": len(games),
        "ppg": avg_pts,
        "apg": avg_ast,
        "rpg": avg_reb,
        "fg_pct": avg_fg,
        "fg3_pct": avg_fg3,
        "wins": int(wins),
        "losses": int(losses),
        "last_5_games": last5.to_dict('records')
    }
def get_similar_players(player_name: str, top_n: int = 5) -> dict:
    """
    Find statistically similar players using cosine similarity.
    Loads pre-computed similarity data on first call.
    """
    import unicodedata
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    from nba_api.stats.endpoints import leaguedashplayerstats

    time.sleep(0.6)

    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season='2024-25',
        per_mode_detailed='PerGame'
    )
    df = player_stats.get_data_frames()[0]

    similarity_features = [
        'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3M', 'FG3_PCT', 'FT_PCT', 'MIN', 'PLUS_MINUS'
    ]

    df_filtered = df[df['MIN'] >= 15].copy()
    df_filtered = df_filtered.dropna(subset=similarity_features)

    def normalize_name(name):
        return ''.join(
            c for c in unicodedata.normalize('NFD', name)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    df_filtered['PLAYER_NAME_NORMALIZED'] = df_filtered['PLAYER_NAME'].apply(normalize_name)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[similarity_features].values)
    similarity_matrix = cosine_similarity(X_scaled)

    search_term = normalize_name(player_name)
    mask = df_filtered['PLAYER_NAME_NORMALIZED'].str.contains(search_term, case=False, na=False)
    matches = df_filtered[mask]

    if len(matches) == 0:
        return {"error": f"Player '{player_name}' not found"}

    player_idx = matches.index[0]
    filtered_positions = list(df_filtered.index)
    pos = filtered_positions.index(player_idx)

    similarity_scores = similarity_matrix[pos]
    sorted_indices = similarity_scores.argsort()[::-1]

    similar_players = []
    for idx in sorted_indices[1:top_n+1]:
        p = df_filtered.iloc[idx]
        similar_players.append({
            "name": p['PLAYER_NAME'],
            "team": p['TEAM_ABBREVIATION'],
            "similarity_score": round(float(similarity_scores[idx]), 3),
            "pts": round(float(p['PTS']), 1),
            "ast": round(float(p['AST']), 1),
            "reb": round(float(p['REB']), 1),
            "fg_pct": round(float(p['FG_PCT']) * 100, 1)
        })

    target = df_filtered.iloc[pos]
    return {
        "player": target['PLAYER_NAME'],
        "stats": {
            "pts": round(float(target['PTS']), 1),
            "ast": round(float(target['AST']), 1),
            "reb": round(float(target['REB']), 1),
            "fg_pct": round(float(target['FG_PCT']) * 100, 1)
        },
        "similar_players": similar_players
    }