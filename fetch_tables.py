import pandas as pd

for year in range(21,25):
    for period in range(1,5):
        dfs = []
        for i in range (50):
            file_path = 'html/20{}-Q{}-{}.html'.format(year, period, i)
            df = pd.read_html(file_path)
            dfs.append(df[2])
        
        combined = pd.concat(dfs)
        combined.to_csv('csv/20{}-Q{}.csv'.format(year, period), index=False)
    