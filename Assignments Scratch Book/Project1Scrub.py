import pandas as pd
from matplotlib import pyplot as plt
from math import log, e
import numpy as np


def date_parse(s):
    t = s.split(".")
    t = str(":".join(t))
    try:
        return pd.datetime.strptime(t, '%Y%m%d:%H:%M:%S:%f')
    except ValueError:
        return "Error"


def isvalid(row):
    if len(row) == 3 and isinstance(row["Price"],float) and isinstance(row["Units"],int) and row["Timestamp"] != "Error":
        return True
    return False


df = pd.read_csv("data.txt", header=None)
df.columns = ["Timestamp", "Price", "Units"]
df.Price = df.Price.astype(float)
df.Units = df.Units.astype(int)


def segregate(df):
    signal = []
    noise = []
    df.Timestamp = df.Timestamp.apply(lambda x: date_parse(x))
    for i in range(len(df)):
        if not isvalid(df.loc[i]):
            noise.append(i)

    dt = date_parse("00000000:00:00:02.000000")

    print(noise)


segregate(df)
# i = 1
# lt = df.Timestamp[i-1]
# while i <= 100000:
#     if abs(df.Timestamp[i-1] - df.Timestamp[i]) < dt:
#         if df.Timestamp[i-1] > df.Timestamp[i]:
#             df.Timestamp[i], df.Timestamp[i-1] = df.Timestamp[i-1], df.Timestamp[i]
#         signal.append(df.Index[i-1])
#         signal.append(df.Index[i])
#
#     else:
#         if df.Timestamp[i-1] > df.Timestamp[i]:
#             signal.append(df.Index[i-1])



#df.sort_values(by=['Timestamp'], ascending=True)
#df.Price = pd.to_numeric(df.Price, errors='coerce')
#df["Previous_Price"] = df.Price.shift(-1)

#df["Returns"] = df.Price / df.Previous_Price

#df["Log_Returns"] = np.log(df.Returns)
#df["Returns"] = np.log(df.Price) - np.log(df.Price.shift(-1))
#df["Log_Returns"] = np.log(1-df.Returns)

print(df.tail())
bins = np.linspace(-10,10)
# X = [i for i in range(100000)]
# plt.scatter(X,df.Returns)
# plt.ylim((-8, 8))
# plt.show()
X = range(-10, 10, 1)
#ydf = pd.crosstab(X, df.Returns)
# Y = pd.cut(df.Returns,bins)
# print(Y)
#
# plt.scatter(X,Y)
# plt.show()
#mu = df.Returns.mean()
#mu = np.mean(df.Returns)
#sigma = df.Returns.std()

#print(mu, sigma)
#print(np.min(df.Returns))