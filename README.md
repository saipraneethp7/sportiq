# 🏀 SportIQ — NBA Intelligence Platform

An AI-powered NBA analytics platform that combines real-time data, 
machine learning models and AI narrative generation to deliver 
professional-grade sports intelligence.

🔗 **Live Demo:** https://joyful-bombolone-e7a0d8.netlify.app

---

## ✨ Features

- **Real-time Player Stats** — Live season stats, game logs and performance charts for any NBA player
- **ML Game Predictor** — Linear regression model predicts player scoring using opponent defense, rest days, recent form and team momentum
- **Player DNA Engine** — Cosine similarity across 12 statistical dimensions finds the 5 most statistically similar players
- **AI Scouting Reports** — LLaMA 3.3 70B generates professional scouting reports from real stats
- **Head to Head Comparison** — Radar chart and stat bars comparing any two players side by side
- **Season Timeline** — Full season scoring visualization with rolling average and win/loss context
- **Team Defense Rankings** — All 30 NBA teams ranked by defensive rating
- **Player Headshots** — Official NBA headshot images for every player

---

## 🧠 Machine Learning
Model:     Linear Regression (scikit-learn)
Training:  52 NBA games (80/20 train/test split)
Features:  Home/Away, Days rest, Opponent DEF rating,
Recent form (last 5 games), Team momentum
MAE:       ~4.4 points on unseen games

**Feature Engineering (discovered from real data):**
- Home court adds ~1.6 points and 4% FG efficiency
- Team momentum on winning streaks adds ~3.6 PPG
- Opponent defensive rating correlates more with team wins than individual scoring
- LeBron performs better on 1-2 days rest than longer breaks

**Player DNA (Cosine Similarity):**
- Normalizes 12 stats using StandardScaler
- Computes 370x370 similarity matrix across all rotation players
- Finds statistically similar players regardless of era or position

---

## 🛠️ Tech Stack
Data:      nba_api, pandas, numpy
ML:        scikit-learn (LinearRegression, StandardScaler, cosine_similarity)
Backend:   Python, FastAPI, uvicorn
AI:        Groq API (LLaMA 3.3 70B)
Frontend:  HTML, CSS, JavaScript, Chart.js
Deploy:    Render (backend), Netlify (frontend)

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/saipraneethp7/sportiq.git
cd sportiq
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**4. Add your API key**

Create `backend/.env`:
GROQ_API_KEY=your_groq_key_here

**5. Run the backend**
```bash
uvicorn main:app --reload
```

**6. Open the frontend**

Open `frontend/index.html` with Live Server in VS Code.

---

## 📊 API Endpoints
GET  /player/{name}          — Season stats and game logs
POST /predict/{name}         — ML scoring prediction
GET  /similar/{name}         — Player DNA similarity
GET  /scout/{name}           — AI scouting report
GET  /compare                — Head to head comparison
GET  /timeline/{name}        — Full season scoring timeline
GET  /teams/defense          — Team defensive rankings

---

## 📁 Project Structure
sportiq/
├── backend/
│   ├── main.py              — FastAPI routes
│   ├── data_fetcher.py      — NBA API and feature engineering
│   ├── predictor.py         — ML model inference
│   └── requirements.txt
├── frontend/
│   ├── index.html           — Landing page
│   └── app.html             — Analytics dashboard
├── models/
│   ├── lebron_scorer.pkl    — Trained ML model
│   └── feature_columns.pkl — Feature names
├── notebooks/
│   ├── explore.ipynb        — Data exploration
│   ├── model.ipynb          — ML model training
│   └── similarity.ipynb     — Similarity engine
└── README.md

---

## 👤 Author

**Sai Praneeth**
- GitHub: [@saipraneethp7](https://github.com/saipraneethp7)
- University: UMKC, CS Class of 2026

---

## 📄 License

MIT License