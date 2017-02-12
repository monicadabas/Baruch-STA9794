import datetime

# Q1. When will all the bitcoins be mined

def Q1():
    total = 0 # number of bitcoins mined
    n = 0 # number of times reward is halved
    bitcoins = 21*(10**6) # total bitcoins to be mined
    reward = 50
    while not(total >= bitcoins):
        total += reward*(1/(2**(n))) * 21 * (10**4)
        n += 1
        print("total: {}, n: {}".format(total,n))

    total_minutes = n * 21 * 10**5 # minutes required to mine all blocks
    print(total, n, total_minutes)

    past_time = datetime.datetime(2009, month = 1, day = 3, hour = 18, minute = 15, second = 5)

    t = past_time + datetime.timedelta(minutes = total_minutes)
    print(t)

Q1()