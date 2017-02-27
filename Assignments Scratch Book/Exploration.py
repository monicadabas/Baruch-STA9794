import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



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

#print(min(df.Price), min(df.Units))

# Y = []
# ts_min = date_parse("00010101:00:00:00.000000")
# y = 0
#
# for i in range(len(df)):
#     t = date_parse(df.loc[i].Timestamp)
#
#     if not isinstance(t, str) and (abs(t-ts_min).total_seconds() <= 3 or t>ts_min):
#         if t>ts_min:
#             ts_min = t
#         continue
#
#     else:
#
#         Y.append(i)
# print(len(Y))
# print(Y)
df.Price = df.Price.apply(lambda x: x/100)
plt.hist(df.Price, bins=np.arange(-10, 30))
plt.show()