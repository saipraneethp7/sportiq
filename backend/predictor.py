import joblib
import numpy as np
import os

# Load the trained model we saved on Day 3
# os.path makes sure the path works regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lebron_scorer.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_columns.pkl')

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

def predict_player_score(
    is_home: int,
    days_rest: float,
    opp_def_rating: float,
    recent_form: float,
    win_rate: float
) -> dict:
    """
    Load our trained model and make a prediction.
    
    This function is the bridge between our Jupyter notebook work
    and the real world API. Same logic, now production ready.
    """

    # Build the feature array in the exact same order
    # the model was trained on. Order matters in ML.
    features = np.array([[
        is_home,
        days_rest,
        opp_def_rating,
        recent_form,
        win_rate
    ]])

    # Make the prediction
    predicted_points = model.predict(features)[0]

    # Calculate a confidence range
    # Our model has MAE of 4.4 so we add/subtract that for a range
    margin = 4.4
    low = round(max(0, predicted_points - margin), 1)
    high = round(predicted_points + margin, 1)

    return {
        "predicted_points": round(float(predicted_points), 1),
        "range_low": low,
        "range_high": high,
        "confidence_note": f"Model predicts within ~{margin} points on average"
    }