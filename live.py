import os
import signal
import time
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, date
from nba_api.live.nba.endpoints import scoreboard, boxscore

# --- Load models and scaler ---

scaler = joblib.load('models/scaler.joblib')
regressor = joblib.load('models/regressor.joblib')
classifier = joblib.load('models/classifier.joblib')

# Feature columns must match train.py exactly
FEATURE_COLS = [
    '(H)PTS', '(H)FGM', '(H)FGA', '(H)3PM', '(H)3PA', '(H)FTM', '(H)FTA',
    '(H)OREB', '(H)DREB', '(H)REB', '(H)AST', '(H)TOV', '(H)STL', '(H)BLK', '(H)PF', '(H)+/-',
    '(A)PTS', '(A)FGM', '(A)FGA', '(A)3PM', '(A)3PA', '(A)FTM', '(A)FTA',
    '(A)OREB', '(A)DREB', '(A)REB', '(A)AST', '(A)TOV', '(A)STL', '(A)BLK', '(A)PF', '(A)+/-',
    'score_diff', 'abs_score_diff', 'total_pts_q123', 'pace',
    'h_fg_pct', 'a_fg_pct', 'h_3p_pct', 'a_3p_pct', 'h_ft_pct', 'a_ft_pct',
]

# Training set average Q4 points (for synthetic line fallback)
TRAIN_Q4_AVG = 54.5  # from train.py output

# Load from .env file if present, otherwise fall back to env var
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
POLL_INTERVAL = 60  # seconds

# Telegram bot notifications
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')


# --- Notifications ---

def send_notification(message, title='NBA Monitor'):
    """Send a push notification via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    text = f"*{title}*\n\n{message}"
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': text,
                'parse_mode': 'Markdown',
            },
            timeout=10,
        )
        print(f"  [Telegram] Sent: {title}")
    except Exception as e:
        print(f"  [Telegram] Failed: {e}")


def format_games_summary(games):
    """Format a list of scoreboard games into a readable summary."""
    if not games:
        return "No games scheduled."

    lines = []
    for g in games:
        home = g['homeTeam']['teamTricode']
        away = g['awayTeam']['teamTricode']
        status = g['gameStatus']
        status_text = g.get('gameStatusText', '').strip()

        if status == 1:  # scheduled
            lines.append(f"  {away} @ {home} — {status_text}")
        elif status == 2:  # in progress
            h_score = g['homeTeam'].get('score', 0)
            a_score = g['awayTeam'].get('score', 0)
            lines.append(f"  {away} @ {home} — {a_score}-{h_score} ({status_text})")
        else:  # final
            h_score = g['homeTeam'].get('score', 0)
            a_score = g['awayTeam'].get('score', 0)
            lines.append(f"  {away} @ {home} — {a_score}-{h_score} (Final)")

    return '\n'.join(lines)


# --- Tricode → full team name (for matching Odds API responses) ---

TRICODE_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BKN': 'Brooklyn Nets', 'BOS': 'Boston Celtics',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
}


# --- Odds API helpers ---

# Cache: one API call returns all live games. Reuse within the same poll cycle
# so multiple Q3 endings in one cycle don't burn extra requests.
_odds_cache = {'data': {}, 'poll_id': -1}


def fetch_live_odds(poll_id):
    """Fetch live NBA totals from The Odds API. Returns dict of {(home, away): total}.
    Results are cached per poll cycle — multiple calls in the same cycle reuse the first result."""
    if not ODDS_API_KEY:
        return {}

    if _odds_cache['poll_id'] == poll_id:
        print(f"  [Odds API] Using cached odds from this poll cycle")
        return _odds_cache['data']

    url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds-live/'
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'totals',
        'oddsFormat': 'american',
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        remaining = resp.headers.get('x-requests-remaining', '?')
        print(f"  [Odds API] requests remaining this month: {remaining}")

        odds = {}
        for event in resp.json():
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            for bm in event.get('bookmakers', []):
                for market in bm.get('markets', []):
                    if market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                odds[(home, away)] = outcome['point']
                                break
                        break
                break  # use first bookmaker

        _odds_cache['data'] = odds
        _odds_cache['poll_id'] = poll_id
        return odds
    except Exception as e:
        print(f"  [Odds API] Error: {e}")
        return {}


def fetch_live_boxscore(game_id):
    """Fetch live box score for a game. Returns (home_stats, away_stats) dicts or None."""
    try:
        box = boxscore.BoxScore(game_id=game_id)
        data = box.get_dict()
        game = data['game']

        home = game['homeTeam']
        away = game['awayTeam']

        def extract_stats(team):
            s = team.get('statistics', {})
            return {
                'TEAM': team['teamTricode'],
                'PTS': s.get('points', 0),
                'FGM': s.get('fieldGoalsMade', 0),
                'FGA': s.get('fieldGoalsAttempted', 0),
                '3PM': s.get('threePointersMade', 0),
                '3PA': s.get('threePointersAttempted', 0),
                'FTM': s.get('freeThrowsMade', 0),
                'FTA': s.get('freeThrowsAttempted', 0),
                'OREB': s.get('reboundsOffensive', 0),
                'DREB': s.get('reboundsDefensive', 0),
                'REB': s.get('reboundsTotal', 0),
                'AST': s.get('assists', 0),
                'TOV': s.get('turnovers', 0),
                'STL': s.get('steals', 0),
                'BLK': s.get('blocks', 0),
                'PF': s.get('foulsPersonal', 0),
                '+/-': s.get('plusMinusPoints', 0),
            }

        return extract_stats(home), extract_stats(away)
    except Exception as e:
        print(f"  [BoxScore] Error for {game_id}: {e}")
        return None


# --- Feature engineering & prediction ---

def build_features(home_stats, away_stats):
    """Convert home/away stats into the 42-feature vector the model expects."""
    raw = {}
    for stat in ['PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA',
                 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', '+/-']:
        raw[f'(H){stat}'] = float(home_stats[stat])
        raw[f'(A){stat}'] = float(away_stats[stat])

    # Derived features
    raw['score_diff'] = raw['(H)PTS'] - raw['(A)PTS']
    raw['abs_score_diff'] = abs(raw['score_diff'])
    raw['total_pts_q123'] = raw['(H)PTS'] + raw['(A)PTS']
    raw['pace'] = raw['(H)FGA'] + raw['(A)FGA']
    raw['h_fg_pct'] = raw['(H)FGM'] / raw['(H)FGA'] if raw['(H)FGA'] > 0 else 0
    raw['a_fg_pct'] = raw['(A)FGM'] / raw['(A)FGA'] if raw['(A)FGA'] > 0 else 0
    raw['h_3p_pct'] = raw['(H)3PM'] / raw['(H)3PA'] if raw['(H)3PA'] > 0 else 0
    raw['a_3p_pct'] = raw['(A)3PM'] / raw['(A)3PA'] if raw['(A)3PA'] > 0 else 0
    raw['h_ft_pct'] = raw['(H)FTM'] / raw['(H)FTA'] if raw['(H)FTA'] > 0 else 0
    raw['a_ft_pct'] = raw['(A)FTM'] / raw['(A)FTA'] if raw['(A)FTA'] > 0 else 0

    return np.array([raw[col] for col in FEATURE_COLS]).reshape(1, -1)


def predict_game(home_stats, away_stats, live_total=None):
    """Run both models and return prediction summary."""
    features = build_features(home_stats, away_stats)
    features_scaled = scaler.transform(features)

    # Regressor: predict total game score
    reg_pred = regressor.predict(features_scaled)[0]

    # Synthetic line (always available)
    q123_total = home_stats['PTS'] + away_stats['PTS']
    synthetic_line = q123_total + TRAIN_Q4_AVG

    # Use live total if available, otherwise synthetic
    line = live_total if live_total else synthetic_line
    line_source = 'live' if live_total else 'synthetic'

    # Classifier: predict over/under (needs posted_total as additional feature)
    clf_features = np.column_stack([features_scaled, np.array([[line]])])
    clf_prob_under = classifier.predict_proba(clf_features)[0][1]

    return {
        'reg_pred': reg_pred,
        'q123_total': q123_total,
        'synthetic_line': synthetic_line,
        'live_total': live_total,
        'line_used': line,
        'line_source': line_source,
        'clf_prob_under': clf_prob_under,
        'model_edge': reg_pred - line,  # negative = model thinks under
    }


def print_prediction(home_team, away_team, home_stats, away_stats, pred):
    """Print a formatted prediction to console."""
    print(f"\n{'='*60}")
    print(f"  {home_team} vs {away_team} — Q3 ENDED")
    print(f"  Q1-Q3 Score: {home_team} {home_stats['PTS']} - {away_stats['PTS']} {away_team}")
    print(f"  Q1-Q3 Total: {pred['q123_total']}")
    print(f"  Score Diff:  {home_stats['PTS'] - away_stats['PTS']:+d}")
    print(f"")
    print(f"  Model predicted total:  {pred['reg_pred']:.1f}")
    print(f"  Synthetic line:         {pred['synthetic_line']:.1f}")
    if pred['live_total']:
        print(f"  Live O/U line:          {pred['live_total']}")
    print(f"  Line used ({pred['line_source']:9s}): {pred['line_used']:.1f}")
    print(f"  P(under):               {pred['clf_prob_under']:.1%}")
    print(f"  Model edge:             {pred['model_edge']:+.1f} pts")
    print(f"")

    is_signal = pred['clf_prob_under'] > 0.60 and pred['model_edge'] < -2
    if is_signal:
        print(f"  >>> SIGNAL: BET UNDER <<<")
    elif pred['clf_prob_under'] > 0.55:
        print(f"  --- Weak under signal (below threshold) ---")
    else:
        print(f"  --- No signal ---")
    print(f"{'='*60}")


def format_q3_notification(home_team, away_team, home_stats, away_stats, pred):
    """Build the Q3-ended notification message with full model results."""
    is_signal = pred['clf_prob_under'] > 0.60 and pred['model_edge'] < -2

    lines = [
        f"{home_team} {home_stats['PTS']} - {away_stats['PTS']} {away_team}",
        f"Q1-Q3 Total: {pred['q123_total']}  Diff: {home_stats['PTS'] - away_stats['PTS']:+d}",
        f"",
        f"Predicted total: {pred['reg_pred']:.1f}",
        f"Line ({pred['line_source']}): {pred['line_used']:.1f}",
        f"P(under): {pred['clf_prob_under']:.0%}",
        f"Edge: {pred['model_edge']:+.1f} pts",
    ]

    if is_signal:
        lines.append(f"\nSIGNAL: BET UNDER")
    elif pred['clf_prob_under'] > 0.55:
        lines.append(f"\nWeak under lean (below threshold)")

    return '\n'.join(lines)


def log_prediction(home_team, away_team, pred):
    """Append prediction to CSV log."""
    row = {
        'timestamp': datetime.now().isoformat(),
        'home': home_team,
        'away': away_team,
        'q123_total': pred['q123_total'],
        'reg_pred': round(pred['reg_pred'], 1),
        'synthetic_line': round(pred['synthetic_line'], 1),
        'live_total': pred['live_total'],
        'line_source': pred['line_source'],
        'clf_prob_under': round(pred['clf_prob_under'], 3),
        'model_edge': round(pred['model_edge'], 1),
        'signal': pred['clf_prob_under'] > 0.60 and pred['model_edge'] < -2,
    }
    df = pd.DataFrame([row])
    path = 'csv/live_predictions.csv'
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# --- Main loop ---

def main():
    if ODDS_API_KEY:
        print(f"Odds API key loaded.")
    else:
        print(f"No ODDS_API_KEY set. Using synthetic line (Q1-Q3 + {TRAIN_Q4_AVG}).")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(f"Telegram notifications enabled.")
    else:
        print(f"Telegram not configured. Notifications disabled.")

    # --- 1. Startup notification with today's games ---
    try:
        sb = scoreboard.ScoreBoard()
        games = sb.get_dict()['scoreboard']['games']
        summary = format_games_summary(games)
        n_games = len(games)

        startup_msg = f"Monitoring {n_games} game(s) today\n\n{summary}"
        print(f"\n{startup_msg}\n")
        send_notification(startup_msg, title='Service Started')
    except Exception as e:
        print(f"  Error fetching initial scoreboard: {e}")
        games = []
        send_notification("Service started (could not fetch today's games)",
                          title='Service Started')

    game_periods = {}   # game_id -> last known period
    game_statuses = {}  # game_id -> last known gameStatus
    processed = set()   # game_ids we've already predicted
    current_date = date.today()
    poll_id = 0         # increments each cycle, used to cache Odds API calls

    print(f"Polling every {POLL_INTERVAL}s...\n")

    # --- Shutdown handler ---
    def shutdown(signum=None, frame=None):
        send_notification("Service stopped.", title='Service Stopped')
        print("\nStopped.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        try:
            poll_id += 1
            sb = scoreboard.ScoreBoard()
            games = sb.get_dict()['scoreboard']['games']

            # --- 5. New day detection ---
            today = date.today()
            if today != current_date:
                current_date = today
                summary = format_games_summary(games)
                n_games = len(games)
                msg = f"New day — {n_games} game(s) scheduled\n\n{summary}"
                print(f"\n{msg}\n")
                send_notification(msg, title='New Day')

                # Reset tracking for the new day
                game_periods.clear()
                game_statuses.clear()
                processed.clear()

            active = [g for g in games if g['gameStatus'] == 2]
            if active:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(active)} game(s) in progress")

            for game in games:
                game_id = game['gameId']
                period = game['period']
                status = game['gameStatus']
                home_team = game['homeTeam']['teamTricode']
                away_team = game['awayTeam']['teamTricode']

                prev_status = game_statuses.get(game_id, 1)
                prev_period = game_periods.get(game_id, 0)
                game_statuses[game_id] = status
                game_periods[game_id] = period

                # --- 2. Game starting ---
                if prev_status == 1 and status == 2:
                    msg = f"{away_team} @ {home_team}"
                    print(f"  Tipoff: {msg}")
                    send_notification(msg, title='Game Starting')

                # --- 3. Q3 ended — run models and notify ---
                if prev_period <= 3 and period >= 4 and game_id not in processed:
                    processed.add(game_id)
                    print(f"\n  Q3 ended: {home_team} vs {away_team} — fetching data...")

                    result = fetch_live_boxscore(game_id)
                    if not result:
                        continue
                    home_stats, away_stats = result

                    live_total = None
                    if ODDS_API_KEY:
                        odds = fetch_live_odds(poll_id)
                        home_full = TRICODE_TO_NAME.get(home_team, home_team)
                        away_full = TRICODE_TO_NAME.get(away_team, away_team)
                        live_total = odds.get((home_full, away_full))

                    pred = predict_game(home_stats, away_stats, live_total)
                    print_prediction(home_team, away_team, home_stats, away_stats, pred)
                    log_prediction(home_team, away_team, pred)

                    is_signal = pred['clf_prob_under'] > 0.60 and pred['model_edge'] < -2
                    title = f"SIGNAL: {home_team} vs {away_team}" if is_signal else f"Q3 End: {home_team} vs {away_team}"
                    ntfy_msg = format_q3_notification(home_team, away_team, home_stats, away_stats, pred)
                    send_notification(ntfy_msg, title=title)

        except SystemExit:
            break
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
