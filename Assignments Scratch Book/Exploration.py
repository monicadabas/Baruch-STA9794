import pandas as pd
from matplotlib import pyplot as plt
#from math import log, e
import numpy as np

x = 'o'
x = float(x)
print(isinstance(x,float))


def date_parse(s):
    t = s.split(".")
    t = str(":".join(t))
    try:
        return pd.datetime.strptime(t, '%Y%m%d:%H:%M:%S:%f')
    except ValueError:
        return "Error"

df = pd.read_csv("data.txt", header=None)
df.columns = ["Timestamp", "Price", "Units"]
df.Price = df.Price.astype(float)
df.Units = df.Units.astype(int)

print(min(df.Price), min(df.Units))
#df.Timestamp = df.Timestamp.apply(lambda x: date_parse(x))
#t0 = date_parse(df.loc[0].Timestamp)
Y = []
ts_min = date_parse("00010101:00:00:00.000000")
y = 0
#print(type(ts_min), type(t0))
for i in range(len(df)):
    t = date_parse(df.loc[i].Timestamp)
    # print(t)
    # print(ts_min)
    # print(abs(t-ts_min).total_seconds())
    if not isinstance(t, str) and (abs(t-ts_min).total_seconds() <= 3 or t>ts_min):
        if t>ts_min:
            ts_min = t
        continue
        # y += 1
        # Y.append(y)
        # if t > ts_min:
        #     ts_min = t
    else:
        #y -= 1
        Y.append(i)
print(len(Y))
print(Y)
