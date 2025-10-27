import pandas as pd
import numpy as np

for year in range(21,25):
    for period in range(1,5):
        data = {}
        df = pd.read_csv('csv/20{}-Q{}.csv'.format(year, period))
        column_names = df.columns
        new_column_names = ['(H)' + c for c in column_names] + ['(A)' + c for c in column_names]

        for index, row in df.iterrows():
            away = '@' in row['MATCH UP']
            if away:
                teams = row['MATCH UP'].split(' @ ')
                key = teams[1] + '-' + teams[0] + "-" + row['GAME DATE']
            else:
                teams = row['MATCH UP'].split(' vs. ')
                key = teams[0] + '-' + teams[1] + "-" + row['GAME DATE']

            if key in data:
                data[key] = [key] + data[key] + row.tolist() if away else [key] + row.tolist() + data[key]
            else:
                data[key] = row.tolist()
        
        new_column_names = ['KEY'] + new_column_names
        combined = pd.DataFrame(data.values(), columns=new_column_names)
        combined.to_csv('csv/20{}-Q{}-combined.csv'.format(year, period), index=False)

