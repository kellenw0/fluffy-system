import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier

# --- Step A: Load and split ---

df = pd.read_csv('csv/dataset.csv')

FEATURE_COLS = [
    # Raw Q1-Q3 stats (32)
    '(H)PTS', '(H)FGM', '(H)FGA', '(H)3PM', '(H)3PA', '(H)FTM', '(H)FTA',
    '(H)OREB', '(H)DREB', '(H)REB', '(H)AST', '(H)TOV', '(H)STL', '(H)BLK', '(H)PF', '(H)+/-',
    '(A)PTS', '(A)FGM', '(A)FGA', '(A)3PM', '(A)3PA', '(A)FTM', '(A)FTA',
    '(A)OREB', '(A)DREB', '(A)REB', '(A)AST', '(A)TOV', '(A)STL', '(A)BLK', '(A)PF', '(A)+/-',
    # Derived features (10)
    'score_diff', 'abs_score_diff', 'total_pts_q123', 'pace',
    'h_fg_pct', 'a_fg_pct', 'h_3p_pct', 'a_3p_pct', 'h_ft_pct', 'a_ft_pct',
]

# Temporal split: train on earlier seasons, test on most recent
seasons = sorted(df['season'].unique())
test_season = seasons[-1]
train_seasons = seasons[:-1]

print(f"Train seasons: {train_seasons}")
print(f"Test season:   {test_season}")

train_df = df[df['season'].isin(train_seasons)]
test_df = df[df['season'] == test_season]

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

X_train = train_df[FEATURE_COLS].values
X_test = test_df[FEATURE_COLS].values
y_train_reg = train_df['total_score'].values
y_test_reg = test_df['total_score'].values

# --- Step B: Normalize ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.joblib')

# --- Step C: Baseline ---

# Naive baseline: Q1-Q3 total points + average Q4 scoring from training set
train_q4_avg = (train_df['total_score'] - train_df['total_pts_q123']).mean()
baseline_preds = test_df['total_pts_q123'].values + train_q4_avg

baseline_mae = mean_absolute_error(y_test_reg, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test_reg, baseline_preds))

print(f"\n{'='*60}")
print(f"BASELINE (Q1-Q3 pts + avg Q4 = {train_q4_avg:.1f} pts)")
print(f"  MAE:  {baseline_mae:.2f}")
print(f"  RMSE: {baseline_rmse:.2f}")

# --- Step D: XGBoost Regressor ---

reg = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
reg.fit(X_train_scaled, y_train_reg)

reg_preds = reg.predict(X_test_scaled)
reg_mae = mean_absolute_error(y_test_reg, reg_preds)
reg_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))

print(f"\n{'='*60}")
print(f"XGBOOST REGRESSOR")
print(f"  MAE:  {reg_mae:.2f} (baseline: {baseline_mae:.2f}, delta: {baseline_mae - reg_mae:+.2f})")
print(f"  RMSE: {reg_rmse:.2f} (baseline: {baseline_rmse:.2f}, delta: {baseline_rmse - reg_rmse:+.2f})")

# Feature importance
importances = reg.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
print(f"\n  Top 15 features:")
for name, imp in feat_imp[:15]:
    print(f"    {name:20s} {imp:.4f}")

joblib.dump(reg, 'models/regressor.joblib')

# --- Step E: XGBoost Classifier ---

# Classifier uses features + posted_total as input
clf_feature_cols = FEATURE_COLS + ['posted_total']

# Exclude rows where under is NaN (pushes)
train_clf = train_df.dropna(subset=['under'])
test_clf = test_df.dropna(subset=['under'])

X_train_clf = train_clf[FEATURE_COLS].values
X_test_clf = test_clf[FEATURE_COLS].values

# Scale with same scaler, then append posted_total (unscaled — it's on a different scale)
X_train_clf_scaled = np.column_stack([
    scaler.transform(X_train_clf),
    train_clf['posted_total'].values.reshape(-1, 1)
])
X_test_clf_scaled = np.column_stack([
    scaler.transform(X_test_clf),
    test_clf['posted_total'].values.reshape(-1, 1)
])

y_train_clf = train_clf['under'].astype(int).values
y_test_clf = test_clf['under'].astype(int).values

clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
clf.fit(X_train_clf_scaled, y_train_clf)

clf_preds = clf.predict(X_test_clf_scaled)
clf_probs = clf.predict_proba(X_test_clf_scaled)[:, 1]  # P(under)

acc = accuracy_score(y_test_clf, clf_preds)
prec = precision_score(y_test_clf, clf_preds)
rec = recall_score(y_test_clf, clf_preds)
f1 = f1_score(y_test_clf, clf_preds)
auc = roc_auc_score(y_test_clf, clf_probs)

print(f"\n{'='*60}")
print(f"XGBOOST CLASSIFIER (predict under)")
print(f"  Accuracy:  {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")
print(f"  F1:        {f1:.3f}")
print(f"  ROC AUC:   {auc:.3f}")
print(f"  Base rate:  {y_test_clf.mean():.3f} (under rate in test set)")

# Feature importance
clf_importances = clf.feature_importances_
clf_feat_imp = sorted(zip(clf_feature_cols, clf_importances), key=lambda x: x[1], reverse=True)
print(f"\n  Top 15 features:")
for name, imp in clf_feat_imp[:15]:
    print(f"    {name:20s} {imp:.4f}")

joblib.dump(clf, 'models/classifier.joblib')

# --- Step F: Simulated betting ---

print(f"\n{'='*60}")
print(f"SIMULATED BETTING (test set)")

# Strategy 1: Bet under whenever classifier predicts under
BET_SIZE = 100
WIN_PAYOUT = 90.91  # -110 odds

under_bets = clf_preds == 1
n_bets = under_bets.sum()
wins = (clf_preds[under_bets] == y_test_clf[under_bets]).sum()
losses = n_bets - wins
profit = wins * WIN_PAYOUT - losses * BET_SIZE

print(f"\n  Strategy 1: Classifier predicts under → bet under")
print(f"    Bets placed: {n_bets}")
print(f"    Wins: {wins}, Losses: {losses}")
print(f"    Win rate: {wins/n_bets:.3f}" if n_bets > 0 else "    Win rate: N/A")
print(f"    Profit: ${profit:+.2f}")
print(f"    ROI: {profit/(n_bets*BET_SIZE)*100:+.1f}%" if n_bets > 0 else "    ROI: N/A")

# Strategy 2: Combined signal — regressor predicts N+ points below line AND classifier confident
print(f"\n  Strategy 2: Combined signal (regressor < line AND classifier > 60% confident)")

# Get regressor predictions for the classifier test set
reg_preds_clf = reg.predict(scaler.transform(test_clf[FEATURE_COLS].values))
posted_totals = test_clf['posted_total'].values

for margin in [2, 3, 5]:
    combined_mask = (reg_preds_clf < posted_totals - margin) & (clf_probs > 0.60)
    n_combined = combined_mask.sum()
    if n_combined > 0:
        combined_wins = (y_test_clf[combined_mask] == 1).sum()
        combined_losses = n_combined - combined_wins
        combined_profit = combined_wins * WIN_PAYOUT - combined_losses * BET_SIZE
        print(f"\n    Margin >= {margin} pts:")
        print(f"      Bets: {n_combined}, Wins: {combined_wins}, Losses: {combined_losses}")
        print(f"      Win rate: {combined_wins/n_combined:.3f}")
        print(f"      Profit: ${combined_profit:+.2f}")
        print(f"      ROI: {combined_profit/(n_combined*BET_SIZE)*100:+.1f}%")
    else:
        print(f"\n    Margin >= {margin} pts: No bets triggered")

# Strategy 3: Synthetic live line — how much edge survives after accounting for
# what's already obvious from Q1-Q3 scoring pace?
#
# The pre-game line implies an expected Q4: expected_q4 = posted_total - total_pts_q123
# A naive live line at end of Q3 would be: total_pts_q123 + expected_q4
#   (which is just posted_total — no edge by definition)
# A smarter synthetic live line adjusts for observed pace:
#   synthetic_live = total_pts_q123 + avg_q4 (from training set)
# This is what a simple model would set the line to after Q3.
# The model's edge = can it predict Q4 scoring better than the training average?
# Only bet when the model predicts below this synthetic line.

print(f"\n  Strategy 3: Synthetic live line (model must beat pace-adjusted expectation)")
print(f"  (synthetic_live = Q1-Q3 actual + training avg Q4 = Q1-Q3 + {train_q4_avg:.1f})")

q123_test = test_clf['total_pts_q123'].values
synthetic_live = q123_test + train_q4_avg
actual_totals = test_clf['total_score'].values

# Sanity check: how often does the actual total come in under the synthetic live line?
synthetic_under_rate = (actual_totals < synthetic_live).mean()
print(f"  Actual under rate vs synthetic line: {synthetic_under_rate:.3f}")

for margin in [0, 1, 2, 3, 5]:
    # Model predicts total below synthetic live line by at least `margin` points
    # AND classifier is confident it's an under
    model_edge = synthetic_live - reg_preds_clf  # positive = model thinks scoring will be suppressed
    synth_mask = (model_edge >= margin) & (clf_probs > 0.60)
    n_synth = synth_mask.sum()
    if n_synth > 0:
        # But the BET is still against the pre-game posted total (that's what you'd bet on)
        synth_wins = (actual_totals[synth_mask] < posted_totals[synth_mask]).sum()
        synth_losses = n_synth - synth_wins
        synth_profit = synth_wins * WIN_PAYOUT - synth_losses * BET_SIZE
        print(f"\n    Model edge >= {margin} pts vs synthetic line:")
        print(f"      Bets: {n_synth}, Wins: {synth_wins}, Losses: {synth_losses}")
        print(f"      Win rate: {synth_wins/n_synth:.3f}")
        print(f"      Profit: ${synth_profit:+.2f}")
        print(f"      ROI: {synth_profit/(n_synth*BET_SIZE)*100:+.1f}%")
    else:
        print(f"\n    Model edge >= {margin} pts vs synthetic line: No bets triggered")

print(f"\nModels saved to models/")
