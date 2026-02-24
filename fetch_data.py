import pandas as pd
import time
from nba_api.stats.endpoints import teamgamelogs

# Column mapping: nba_api names → format expected by build_dataset.py
COL_RENAME = {
    'TEAM_ABBREVIATION': 'TEAM',
    'MATCHUP': 'MATCH UP',
    'GAME_DATE': 'GAME DATE',
    'WL': 'W/L',
    'MIN': 'MIN',
    'FG_PCT': 'FG%',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FT_PCT': 'FT%',
    'PLUS_MINUS': '+/-',
}

# Columns to keep, in order (matching the scraped CSV format)
KEEP_COLS = ['TEAM', 'MATCH UP', 'GAME DATE', 'W/L', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%',
             '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST',
             'TOV', 'STL', 'BLK', 'PF', '+/-']

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']

for season in SEASONS:
    year = season[:4]  # '2021' from '2021-22'
    short_year = year[2:]  # '21'

    for period in range(1, 5):
        print(f"Fetching {season} Q{period}...", end=' ', flush=True)

        logs = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            period_nullable=period,
            season_type_nullable='Regular Season',
        )
        df = logs.get_data_frames()[0]
        print(f"{len(df)} rows")

        # Rename columns to match expected format
        df = df.rename(columns=COL_RENAME)

        # Convert date from ISO format (2022-04-10T00:00:00) to MM/DD/YYYY
        df['GAME DATE'] = pd.to_datetime(df['GAME DATE']).dt.strftime('%m/%d/%Y')

        df = df[KEEP_COLS]

        # Combine home and away rows into single game rows
        data = {}
        column_names = df.columns
        new_column_names = ['(H)' + c for c in column_names] + ['(A)' + c for c in column_names]

        for _, row in df.iterrows():
            matchup = row['MATCH UP']
            away = '@' in matchup

            if away:
                teams = matchup.split(' @ ')
                key = teams[1] + '-' + teams[0] + '-' + row['GAME DATE']
            else:
                teams = matchup.split(' vs. ')
                key = teams[0] + '-' + teams[1] + '-' + row['GAME DATE']

            if key in data:
                if away:
                    data[key] = [key] + data[key] + row.tolist()
                else:
                    data[key] = [key] + row.tolist() + data[key]
            else:
                data[key] = row.tolist()

        new_column_names = ['KEY'] + new_column_names
        combined = pd.DataFrame(data.values(), columns=new_column_names)

        out_path = f'csv/20{short_year}-Q{period}-combined.csv'
        combined.to_csv(out_path, index=False)
        print(f"  → {out_path} ({len(combined)} games)")

        time.sleep(1)  # rate limiting

print("\nDone. All combined CSVs written to csv/")
