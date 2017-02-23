import pandas as pd
from matplotlib import pyplot as plt
from math import log, e
import numpy as np
import logging
from operator import itemgetter


logging.basicConfig(filename ="Project1Scrub_log.log", level= logging.DEBUG, filemode='w')


def date_parse(s):
    t = s.split(".")
    t = str(":".join(t))
    try:
        return pd.datetime.strptime(t, '%Y%m%d:%H:%M:%S:%f')
    except ValueError:
        return "Error"


class IsValidResult:
    def __init__(self, first, second):
        self.first = first
        self.second = second


def isvalid(row,t):
    t0 = date_parse("00010101:00:00:00.000000")
    if len(row) == 3 and isinstance(row[1],float) and row[1]>0 and isinstance(row[2],np.int64) and row[2]>0 and row[0] != "Error":
        if t < row[0]:
            if not((row[0]-t).total_seconds() > 3) or t == t0:
                return IsValidResult(True, row[0])
            else:
                logging.info("Timestamps is more than 3 seconds than earlier timestamp")
                return IsValidResult(False,t)

        elif t == row[0]:
            logging.info("Duplicate timestamp")
            return IsValidResult(False,t)
        else:
            if -3 < (t-row[0]).total_seconds() < 3:
                return IsValidResult(True, t)
            else:
                logging.info("Timestamp is more than 3 seconds earlier")
                return IsValidResult(False, t)
    else:
        if row[0] == "Error":
            logging.info("Incorrect timestamp format")
            return IsValidResult(False,max(t,row[0]))
        else:
            logging.info("Initial condition of datatypes and range not met")
            return IsValidResult(False,t)


df = pd.read_csv("data.txt", header=None)
df.columns = ["Timestamp", "Price", "Units"]
df.Price = df.Price.astype(float)
df.Units = df.Units.astype(int)


def segregate(df):
    signal = list()
    noise = list()
    t = date_parse("00010101:00:00:00.000000")
    for i in range(len(df)):
        logging.info("Row number: {}".format(i))
        row = list()
        row.append(date_parse(df.loc[i].Timestamp))
        row.append(df.loc[i].Price)
        row.append(df.loc[i].Units)

        result = isvalid(row, t)

        if not result.first:
            logging.info("Unit and Datatype: {}, {}".format(row[2],type(row[2])))
            logging.info("Length: {}".format(len(row) == 3))
            logging.info("Is Price float: {}".format(isinstance(row[1],float)))
            logging.info("Is price greater than 0: {}".format(row[1]>0))
            logging.info("Is Unit int: {}".format(isinstance(row[2],np.int64)))
            logging.info("Is Unit greater than 0:{}".format(row[2]>0))
            logging.info("Is Timestamp correct: {}".format(row[0] != "Error"))
            noise.append(i)

        else:
            t = result.second
            signal.append([date_parse(df.loc[i].Timestamp),df.loc[i].Price])

    print(len(noise))
    print("Signal Length: {}".format(len(signal)))
    return signal


signal_Prices = segregate(df)

signal_Prices = sorted(signal_Prices, key=itemgetter(0))
print("Sorted signal length: {}".format(len(signal_Prices)))
dups = []
for i in range(1,len(signal_Prices)):
    if signal_Prices[i][0] == signal_Prices[i-1][0]:
        dups.append(signal_Prices[i][0])

print("Length of dups: {}".format(len(dups)))

print("Signal Prices printing")

i = 0
while i < 1:
    print("time in signal:{}".format(signal_Prices[i][0]))
    print("time in dups: {}".format(dups[i]))
    i += 1

for i in signal_Prices:
    if i[0] in dups:
        del i

print("Sorted and no duplicates signal length: {}".format(len(signal_Prices)))
returns = []
for i in range(1,len(signal_Prices)):
    ret = log(signal_Prices[i][1]/ signal_Prices[i-1][1], e)
    returns.append(ret*100)

print(len(returns), max(returns), min(returns))

plt.hist(returns, bins=np.arange(-150, 150))
plt.show()

