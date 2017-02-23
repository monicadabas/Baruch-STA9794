from __future__ import division
import datetime

# Q1. When will all the bitcoins be mined


def Q1():
    total = 0 # number of bitcoins mined
    n = 0 # number of times reward is halved
    bitcoins = 21*(10**6) # total bitcoins to be mined
    reward = 50

    # in every cycle 210000 blocks are mined which means atleast 210000 satoshis should be added in each cycle
    # reward halving cycle runs till bitcoins added are not less than 210000 satoshis
    # cycle must run only till number of bitcoins mined are less than equal to the maximum bitcoins

    while not(total >= bitcoins):
        bitcoins_added = round(reward*(1/(2**(n))) * 21 * (10**4), 8)
        # print(bitcoins_added)
        if bitcoins_added >= 21 * (10**(-4)):
            total += bitcoins_added
            n += 1
        else:
            break
        # print("total: {}, n: {}".format(total,n))

    total_minutes = n * 21 * 10**5 # minutes required to mine all blocks
    # print(total, n, total_minutes)

    past_time = datetime.datetime(2009, month = 1, day = 3, hour = 18, minute = 15, second = 5)

    t = past_time + datetime.timedelta(minutes = total_minutes)
    print("All bitcoins will be mined by {}".format(t))
    print("All bitcoins will be mined by year {}.".format(t.year))

Q1()

# Output
# All bitcoins will be mined by 2140-10-08 18:15:05
# All bitcoins will be mined by year 2140.



