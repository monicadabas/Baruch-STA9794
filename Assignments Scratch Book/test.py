import pandas as pd

dict = {'a': ['1','2','3'],'b': ['4','s','6']}

df = pd.DataFrame(dict)
df.a = df.a.astype(float)
df.b = df.b.astype(int)
print(df)