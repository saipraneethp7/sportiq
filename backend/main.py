import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from data_fetcher import (
    find_player,
    get_player_stats_summary,
    get_current_features,
    get_team_defense_ratings,
    get_similar_players,
    get_player_game_logs,
    ABBREV_TO_NAME
)
from predictor import predict_player_score

load_dotenv()

app = FastAPI(
    title="SportIQ API",
    description="AI-powered NBA analytics and prediction engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "SportIQ API is running",
        "version": "1.0.0",
        "endpoints": ["/player/{name}", "/predict/{name}", "/teams/defense", "/scout/{name}"]
    }

# ─────────────────────────────────────────────
# Player stats endpoint
# ─────────────────────────────────────────────

@app.get("/player/{name}")
def get_player(name: str):
    """
    Get a player's season stats summary.
    Example: /player/LeBron James
    """
    player = find_player(name)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")

    try:
        stats = get_player_stats_summary(player['id'])
        return {
            "player_id": player['id'],
            "name": player['full_name'],
            "is_active": player['is_active'],
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# Prediction endpoint
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    opponent: str
    is_home: int
    days_rest: int = 2

@app.post("/predict/{name}")
def predict_player(name: str, request: PredictRequest):
    """
    Predict a player's scoring in an upcoming game.
    Example: POST /predict/LeBron James
    Body: {"opponent": "OKC", "is_home": 1, "days_rest": 2}
    """
    player = find_player(name)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")

    try:
        # Get current features for this player
        current = get_current_features(player['id'])

        # Get opponent defensive rating
        defense_ratings = get_team_defense_ratings()
        opponent_full = ABBREV_TO_NAME.get(request.opponent.upper())
        if not opponent_full:
            raise HTTPException(status_code=400, detail=f"Unknown team: {request.opponent}")

        opp_def_rating = defense_ratings.get(opponent_full, 113.7)

        # Make prediction
        prediction = predict_player_score(
            is_home=request.is_home,
            days_rest=request.days_rest,
            opp_def_rating=opp_def_rating,
            recent_form=current['recent_form'],
            win_rate=current['win_rate_5']
        )

        return {
            "player": player['full_name'],
            "opponent": opponent_full,
            "is_home": bool(request.is_home),
            "days_rest": request.days_rest,
            "current_form": current['recent_form'],
            "team_momentum": current['win_rate_5'],
            "opponent_def_rating": opp_def_rating,
            "prediction": prediction
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# Team defense endpoint
# ─────────────────────────────────────────────

@app.get("/teams/defense")
def get_team_defense():
    """Get defensive ratings for all 30 NBA teams ranked best to worst."""
    try:
        ratings = get_team_defense_ratings()
        ranked = sorted(ratings.items(), key=lambda x: x[1])
        return {
            "rankings": [
                {"rank": i+1, "team": team, "def_rating": round(rating, 1)}
                for i, (team, rating) in enumerate(ranked)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{name}")
def get_similar(name: str):
    """
    Find the 5 most statistically similar players to any NBA player.
    Uses cosine similarity across 12 statistical dimensions.
    Example: /similar/LeBron James
    """
    try:
        result = get_similar_players(name)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/compare")
def compare_players(player1: str, player2: str):
    """
    Compare two players head to head.
    Example: /compare?player1=LeBron James&player2=Kevin Durant
    """
    try:
        p1 = find_player(player1)
        p2 = find_player(player2)

        if not p1:
            raise HTTPException(status_code=404, detail=f"Player '{player1}' not found")
        if not p2:
            raise HTTPException(status_code=404, detail=f"Player '{player2}' not found")

        stats1 = get_player_stats_summary(p1['id'])
        stats2 = get_player_stats_summary(p2['id'])

        return {
            "player1": {
                "name": p1['full_name'],
                "player_id": p1['id'],
                "stats": stats1
            },
            "player2": {
                "name": p2['full_name'],
                "player_id": p2['id'],
                "stats": stats2
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
# ─────────────────────────────────────────────
# AI Scouting report endpoint
# ─────────────────────────────────────────────
@app.get("/timeline/{name}")
def get_season_timeline(name: str):
    """
    Get a player's scoring journey across the entire season.
    Returns game by game data for timeline visualization.
    """
    player = find_player(name)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")

    try:
        games = get_player_game_logs(player['id'])
        games_sorted = games.sort_values('GAME_DATE')

        timeline = []
        rolling_avg = games_sorted['PTS'].rolling(5).mean()

        for i, (_, game) in enumerate(games_sorted.iterrows()):
            timeline.append({
                "game_num": i + 1,
                "date": game['GAME_DATE'].strftime('%b %d'),
                "pts": int(game['PTS']),
                "rolling_avg": round(float(rolling_avg.iloc[i]), 1) if i >= 4 else None,
                "matchup": game['MATCHUP'],
                "wl": game['WL']
            })

        return {
            "player": player['full_name'],
            "player_id": player['id'],
            "timeline": timeline,
            "season_avg": round(float(games_sorted['PTS'].mean()), 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/scout/{name}")
def generate_scouting_report(name: str):
    """
    Generate an AI scouting report for a player
    using their real season stats and Groq AI.
    """
    player = find_player(name)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")

    try:
        stats = get_player_stats_summary(player['id'])
        current = get_current_features(player['id'])

        prompt = f"""You are a professional NBA scout writing a concise scouting report.
        
Player: {player['full_name']}
Season Stats 2024-25:
- Points per game: {stats['ppg']}
- Assists per game: {stats['apg']}  
- Rebounds per game: {stats['rpg']}
- Field goal %: {stats['fg_pct']}%
- 3-point %: {stats['fg3_pct']}%
- Record: {stats['wins']}W - {stats['losses']}L
- Games played: {stats['games_played']}
- Current form (last 5 games avg): {current['recent_form']} PPG
- Team win rate last 5 games: {current['win_rate_5']*100:.0f}%

Write a 3 paragraph professional scouting report covering:
1. Current season performance and what the numbers tell us
2. Strengths based on the stats
3. Current form and outlook

Be specific, use the actual numbers, sound like a real NBA analyst.
Do not use generic phrases. Make it feel like genuine expert analysis."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert NBA scout and analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "player": player['full_name'],
            "stats": stats,
            "scouting_report": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))