import pandas as pd
import numpy as np

data = {}
labels = {}
# just need to change a/m to percent and compute (combine 4 quarters of same game using key, then calculate averages and put in one table)
column_names = ['(H)PTS', '(H)FGM', '(H)FGA', '(H)3PM', '(H)3PA', '(H)FTM', '(H)FTA', '(H)OREB', '(H)AST', '(H)STL', '(A)PTS', '(A)FGM', '(A)FGA', '(A)3PM', '(A)3PA', '(A)FTM', '(A)FTA', '(A)OREB', '(A)AST', '(A)STL']
for year in range(21,25):
    for period in [1,2,3,4]:
        df = pd.read_csv('csv/20{}-Q{}-combined.csv'.format(year, period))
        for index, row in df.iterrows():
            key = row['KEY']
            if key in data:
                params = data[key]
                label = labels[key]
            else:
                params = np.zeros(20)
                label = 0
            
            if period < 4: # We want the inputs to include the first three quarters of data and the final score
                new_params = [row['(H)PTS'], row['(H)FGM'], row['(H)FGA'], row['(H)3PM'], row['(H)3PA'], row['(H)FTM'], row['(H)FTA'], row['(H)OREB'], row['(H)AST'], row['(H)STL'], row['(A)PTS'], row['(A)FGM'], row['(A)FGA'], row['(A)3PM'], row['(A)3PA'], row['(A)FTM'], row['(A)FTA'], row['(A)OREB'], row['(A)AST'], row['(A)STL']]
                data[key] = params + np.array(new_params)
            labels[key] = label + row['(H)PTS'] + row['(A)PTS']

i = pd.DataFrame(data.values(), columns=column_names)
i.to_csv('csv/inputs.csv', index=False)


l = pd.DataFrame(labels.values(), columns=['Label'])
l.to_csv('csv/labels.csv', index=False)

