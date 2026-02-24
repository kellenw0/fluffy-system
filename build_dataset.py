import pandas as pd
import numpy as np

# --- Step A: Accumulate Q1-Q3 box score stats, Q1-Q4 for labels ---

# All numeric stats we want to accumulate (per team, home and away)
STAT_COLS = ['PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA',
             'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', '+/-']

feature_cols = [f'(H){s}' for s in STAT_COLS] + [f'(A){s}' for s in STAT_COLS]
n_features = len(feature_cols)  # 32

features = {}   # KEY -> np.array of accumulated Q1-Q3 stats
total_pts = {}  # KEY -> total points across all 4 quarters

for year in range(21, 25):
    for period in [1, 2, 3, 4]:
        path = f'csv/20{year}-Q{period}-combined.csv'
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            key = row['KEY']

            # Skip orphaned rows (unpaired games from combine_home_and_away)
            if not isinstance(key, str) or key.count('-') < 2:
                continue

            if key not in features:
                features[key] = np.zeros(n_features)
                total_pts[key] = 0.0

            # Accumulate total points from all 4 quarters (for regression label)
            total_pts[key] += row['(H)PTS'] + row['(A)PTS']

            # Only accumulate detailed stats from Q1-Q3 (features)
            if period <= 3:
                vals = [row[c] for c in feature_cols]
                features[key] += np.array(vals)

# Build raw features dataframe
raw_df = pd.DataFrame.from_dict(features, orient='index', columns=feature_cols)
raw_df['total_score'] = pd.Series(total_pts)
raw_df.index.name = 'KEY'
raw_df = raw_df.reset_index()

# --- Step B: Derive features ---

raw_df['score_diff'] = raw_df['(H)PTS'] - raw_df['(A)PTS']
raw_df['abs_score_diff'] = raw_df['score_diff'].abs()
raw_df['total_pts_q123'] = raw_df['(H)PTS'] + raw_df['(A)PTS']
raw_df['pace'] = raw_df['(H)FGA'] + raw_df['(A)FGA']
raw_df['h_fg_pct'] = raw_df['(H)FGM'] / raw_df['(H)FGA'].replace(0, np.nan)
raw_df['a_fg_pct'] = raw_df['(A)FGM'] / raw_df['(A)FGA'].replace(0, np.nan)
raw_df['h_3p_pct'] = raw_df['(H)3PM'] / raw_df['(H)3PA'].replace(0, np.nan)
raw_df['a_3p_pct'] = raw_df['(A)3PM'] / raw_df['(A)3PA'].replace(0, np.nan)
raw_df['h_ft_pct'] = raw_df['(H)FTM'] / raw_df['(H)FTA'].replace(0, np.nan)
raw_df['a_ft_pct'] = raw_df['(A)FTM'] / raw_df['(A)FTA'].replace(0, np.nan)

# --- Step C: Join Kaggle betting data ---

kaggle = pd.read_csv('csv/kaggle_betting.csv')

# Normalize team abbreviations to uppercase + fix known mismatches
TEAM_MAP = {
    'sa': 'SAS', 'gs': 'GSW', 'no': 'NOP', 'ny': 'NYK',
    'utah': 'UTA', 'wsh': 'WAS',
}
kaggle['home'] = kaggle['home'].apply(lambda x: TEAM_MAP.get(x, x.upper()))
kaggle['away'] = kaggle['away'].apply(lambda x: TEAM_MAP.get(x, x.upper()))

# Convert Kaggle date (YYYY-MM-DD) to match NBA.com format (MM/DD/YYYY)
kaggle['date'] = pd.to_datetime(kaggle['date'])
kaggle['date_str'] = kaggle['date'].dt.strftime('%m/%d/%Y')

# Build join key: HOME-AWAY-DATE
kaggle['KEY'] = kaggle['home'] + '-' + kaggle['away'] + '-' + kaggle['date_str']

# Check for team abbreviation mismatches before joining
nba_teams = set()
for key in raw_df['KEY']:
    parts = key.split('-')
    nba_teams.add(parts[0])
    nba_teams.add(parts[1])

kaggle_teams = set(kaggle['home'].unique()) | set(kaggle['away'].unique())

nba_only = nba_teams - kaggle_teams
kaggle_only = kaggle_teams - nba_teams

if nba_only:
    print(f"Teams in NBA.com data but NOT in Kaggle: {nba_only}")
if kaggle_only:
    # Only show teams that might be relevant (filter to seasons we care about)
    print(f"Teams in Kaggle but NOT in NBA.com: {kaggle_only}")

# Inner join
merged = raw_df.merge(kaggle[['KEY', 'total', 'id_total', 'season', 'ot_home', 'ot_away']],
                       on='KEY', how='inner')

print(f"\nJoin statistics:")
print(f"  NBA.com games:    {len(raw_df)}")
print(f"  Kaggle games:     {len(kaggle)}")
print(f"  Matched games:    {len(merged)}")
print(f"  Unmatched:        {len(raw_df) - len(merged)}")

# --- Step D: Handle edge cases ---

# Exclude overtime games
ot_mask = (merged['ot_home'] > 0) | (merged['ot_away'] > 0)
print(f"  Overtime games:   {ot_mask.sum()} (excluded)")
merged = merged[~ot_mask]

# Exclude pushes from classification target
push_mask = merged['id_total'] == 2
print(f"  Pushes:           {push_mask.sum()} (excluded from classifier)")

# Build classification label: 1 = under, 0 = over
merged['under'] = (merged['id_total'] == 0).astype(int)
# Set pushes to NaN for the classifier (they'll be excluded during training)
merged.loc[push_mask, 'under'] = np.nan

# Rename for clarity
merged = merged.rename(columns={'total': 'posted_total'})

# Drop join/temp columns
merged = merged.drop(columns=['ot_home', 'ot_away', 'id_total'])

# Drop rows with NaN in feature columns
feature_col_list = feature_cols + ['score_diff', 'abs_score_diff', 'total_pts_q123',
                                    'pace', 'h_fg_pct', 'a_fg_pct', 'h_3p_pct',
                                    'a_3p_pct', 'h_ft_pct', 'a_ft_pct']
nan_before = len(merged)
merged = merged.dropna(subset=feature_col_list)
print(f"  NaN rows dropped: {nan_before - len(merged)}")

# --- Step E: Output ---

merged.to_csv('csv/dataset.csv', index=False)

print(f"\nFinal dataset: {len(merged)} games")
print(f"  Seasons: {sorted(merged['season'].unique())}")
print(f"  Features: {len(feature_col_list)}")
print(f"  Under rate: {merged['under'].mean():.3f}")
print(f"Saved to csv/dataset.csv")
