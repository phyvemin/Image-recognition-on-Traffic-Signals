import pandas as pd

data = pd.read_csv('./test.csv')
data2 = pd.read_csv('./archive/Test.csv')
l = []
j = 0
tot = 0
for i in data.iloc[:,0]:
    x = i[:5]
    if data.iloc[:,1][j]==data2['ClassId'][int(x)]:
        tot += 1
    j += 1

print(tot/data.shape[0])


