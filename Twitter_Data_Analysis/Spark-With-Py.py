"""
Command Line arguments:

Please use config-parameters.py to get the following 4 arguments

--num-executors <number of executors requested>
--executor-cores <number of cores with each executor>
--executor-memory <memory of each executor like 1g, 10g>
<number of files to read in each iteration, preferably in multiples of 60>

Command:

spark-submit --master= <cluster URL> --num-executors <number of executors requested>
--executor-cores <number of cores with each executor> --executor-memory <memory of each executor like 1g, 10g>
<python file> <Data Folder with no slash at the end>
<number of files to read in each iteration, preferably in multiples of 60>

For example, I use the below to run locally (my python file and data folder are in same folder)
spark-submit --master=local[*] read_data_test.py TwitterData 240


"""

from __future__ import division
import sys
from datetime import datetime, timedelta
import logging

# declaration of log file

frmt = '%(levelname)s:%(asctime)s:%(message)s'
fn = "Spark-With-Py-log.log"

if len(sys.argv) == 4:
    log_level = sys.argv[3][6:]
    try:
        num_level = getattr(logging, log_level.upper())
    except AttributeError:
        print("Incorrect logging level provided")
        sys.exit()
    if not isinstance(num_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
else:
    num_level = 'ERROR'

logging.basicConfig(filename=fn, format=frmt, datefmt='%m/%d/%Y %I:%M:%S %p', level=num_level, filemode='w')


logging.info("Importing Libraries....")
t0 = datetime.now()

import os, io, copy
import ujson
import fnmatch
import requests
import pandas as pd
from time import sleep
from math import sqrt
from dateutil import tz
from textblob import TextBlob
from statistics import mean, stdev
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
# from pyspark.sql.types import DateType
# from pyspark.sql.functions import udf


t001 = datetime.now()

logging.info("Time taken to import libraries: {}".format((t001-t0).total_seconds()))


def filter_tweet(t):
    stock_list = ["YHOO", "MSFT", "SBUX", 'NVDA', "IBM"] # "microsoft", "ibm", "starbucks", "nvidia", "yahoo",

    # checks if the user is a potential robot: liked many more tweets compared to self tweet
    try:
        robot_cond1 = (t["user"]["favourites_count"] / t["user"]["statuses_count"] < 100)
    except ZeroDivisionError:
        robot_cond1 = (t["user"]["favourites_count"] < 500)

    # checks if the user is a potential robot: follows many more people compared to user's followers

    try:
        robot_cond2 = (t["user"]["followers_count"] / t["user"]["friends_count"] >= 1)
    except ZeroDivisionError:
        robot_cond2 = (t["user"]["followers_count"] >= 100)

    # checks if a tweet is relevant: tweet in english, contains stock ticker, is not a reply to another tweet
    try:
        lang_cond = (t["lang"] == "en")
        text_cond = any(i in (t["text"] or t["entities"]["hashtags"][0]["text"]) for i in stock_list)
        original_tweet_cond = (t["in_reply_to_screen_name"] is None)
    except:
        return False

    conditions = [lang_cond, text_cond, original_tweet_cond, robot_cond1, robot_cond2]

    if all(conditions):
        return True
    else:
        return False


def utc_to_est(s):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    utc = datetime.strptime(s,'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=from_zone)
    central = utc.astimezone(to_zone)
    return central


def get_sentiment(t):
    sentiment_text = TextBlob(t["text"].decode('unicode_escape').encode('ascii','ignore')).sentiment
    text_polarity = sentiment_text.polarity
    text_subjectivity = sentiment_text.subjectivity

    if len(t["hashtags"]) > 0:
        sentiment_hashtag = TextBlob(t["hashtags"].decode('unicode_escape').encode('ascii','ignore')).sentiment
        hashtag_polarity = sentiment_hashtag.polarity
        hashtag_subjectivity = sentiment_hashtag.subjectivity

        if hashtag_polarity != 0:
            polarity = (text_polarity + hashtag_polarity) / 2
        else:
            polarity = text_polarity
        if hashtag_subjectivity != 0:
            subjectivity = (text_subjectivity + hashtag_subjectivity) / 2
        else:
            subjectivity = text_subjectivity

    else:
        polarity = text_polarity
        subjectivity = text_subjectivity

    senti_ment = round(polarity * (1-subjectivity) * t["sentiment_weight"],4)

    return senti_ment


def parse_tweet(t):
    text = t['text'].replace('\n', ' ').replace(",", ' ').encode('utf-8')
    try:
        hashtag = t['entities']['hashtags'][0]['text']
    except:
        hashtag = ''

    stock_list = ["YHOO", "MSFT", "SBUX", 'NVDA', "IBM"]

    related_stock = map(lambda x: 1 if x else 0, [i in (t["text"] or t["hashtags"]) for i in stock_list])

    try:
        retweeted = 1 if t['retweeted'] else 0
    except:
        retweeted = 0

    try:
        favorited = 1 if t['favorited'] else 0
    except:
        favorited = 0

    try:
        following = 1 if t['user']['following'] else 0
    except:
        following = 0

    try:
        protected = 1 if t['user']['protected'] else 0
    except:
        protected = 0

    try:
        verified = 1 if t['user']['verified'] else 0
    except:
        verified = 0

    try:
        favorite_count = t['favorite_count'] / 100
        favorite_count /= (favorite_count + 10)
    except:
        favorite_count = 0

    sentiment_weight = (1 + retweeted + favorited + following + protected + verified + favorite_count)

    created_at = utc_to_est(t['created_at'])

    created_on = created_at.date()
    created_hour = created_at.hour

    created_on = created_on + timedelta(days=1) if created_hour > 15 else created_on

    row = {'sentiment_weight': sentiment_weight,'hashtags': hashtag, 'text': text}

    senti_ment = get_sentiment(row)
    stock_sentiment = [senti_ment*i for i in related_stock]

    row = {'created_on': created_on, 'created_hour': created_hour,'stock_sentiment': stock_sentiment}

    return row


def get_stocks_price_data(start_date_str, end_date_str):

    start_date = start_date_str - timedelta(days=1)
    end_date = end_date_str + timedelta(days=5)

    start_year = str(start_date.year)
    start_month = str(start_date.month -1)
    start_day = str(start_date.day)

    end_year = str(end_date.year)
    end_month = str(end_date.month -1)
    end_day = str(end_date.day)

    stock_prices_df = pd.DataFrame()
    counter = 0

    for stock in ["YHOO", "MSFT", "SBUX", 'NVDA', "IBM"]:
        stock_url = "https://chart.finance.yahoo.com/table.csv?s=" + stock +"&a="+ start_month +"&b="+ start_day +"&c="+ start_year +"&d="+ end_month +"&e="+ end_day +"&f="+ end_year +"&g=d&ignore=.csv"

        try:
            data = requests.get(stock_url).content
        except:
            logging.info("Connection to Yahoo failed. Check your internet connection")
            sys.exit()

        df = pd.read_csv(io.StringIO(data.decode('utf-8')))

        if counter == 0:
            df.drop(df.columns[[1, 2, 3, 4, 5]], axis=1, inplace=True)
            df.rename(columns={'Adj Close':stock}, inplace=True)
            stock_prices_df["Date"] = pd.Series(df['Date'])

        else:
            df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
            df.rename(columns={'Adj Close':stock}, inplace=True)

        stock_prices_df[stock] = pd.Series(df[stock])
        counter += 1
        sleep(5)

    return stock_prices_df.iloc[::-1]


# returns the trading date for a calender date
# if calendar date is a trading day, returns the calendar date
# else returns the next trading date

def get_trading_date(random_date, trading_dates):

    random_date_as_str = str(random_date)

    final_dates = list(filter(lambda date: date >= random_date_as_str, trading_dates))

    final_date = datetime.strptime(final_dates[0],'%Y-%m-%d').date()

    return final_date


# makes trading decision based on the sentiment, calculates the stocks position in portfolio
# after the trading is done
# returns the available cash amount, dollars invested in each stock (portfolio) and
# number of shares for each stock (positions)

def trading(sentiment, prices, portfolio, cash, positions):

    for i in range(5):
        trading_amount = 5000000
        #sell
        if sentiment[i] < 0:
            if sentiment[i] <= -3:
                if portfolio[i] >= trading_amount*3:
                    trading_amount *= 3
                else:
                    trading_amount = portfolio[i]
            elif sentiment[i] <= -1:
                if portfolio[i] >= trading_amount*2:
                    trading_amount *= 2
                else:
                    trading_amount = portfolio[i]
            else:
                if portfolio[i] >= trading_amount:
                    trading_amount *= 1
                else:
                    trading_amount = portfolio[i]
            cash += trading_amount
            portfolio[i] -= trading_amount
            positions[i] -= trading_amount/prices[i]

        # buy else stay
        elif sentiment[i] > 0.3:
            if sentiment[i] >= 3 and cash >= trading_amount*3:
                trading_amount *= 3
            elif sentiment[i] >= 1 and cash >= trading_amount*2:
                trading_amount *= 2
            elif cash >= trading_amount:
                trading_amount *= 1
            else:
                trading_amount = cash
            cash -= trading_amount
            portfolio[i] += trading_amount
            positions[i] += trading_amount/prices[i]

    return portfolio, cash, positions


def main(path, num_of_files):
    logging.info("In main function, setting up spark...\n")
    conf = SparkConf().setAppName("Tweets Analysis for Trading")
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)

    # get the filepath of all files in an array
    files_list = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(path)
                  for f in fnmatch.filter(files, '*.json.bz2')]

    t01 = datetime.now()

    files_to_read = len(files_list)

    logging.info("Number of files to read: {}".format(files_to_read))
    logging.info("Time taken to set up spark and get files list: {}".format((t01-t001).total_seconds()))

    final_date_sentiment_rdd = None

    # On cluster: read and filter one week's data at once (each hr has 60 files so 60files*24hrs*7days = 10080)
    # For local: num_of_files = 240

    last_iteration = int(files_to_read / num_of_files) if files_to_read % num_of_files == 0 \
        else int(files_to_read / num_of_files) + 1
    files_read = 0
    iteration = 0
    min_date = None
    max_date = None

    while files_read < files_to_read:

        iteration += 1

        # index  till which to read files in this iteration

        index_iteration = files_read + min(num_of_files, files_to_read - files_read)

        logging.info("Iteration: {}".format(iteration))
        logging.info("##################################")
        logging.info("Reading data into rdd...\n")

        t1 = datetime.now()

        # read content of all files into rdd of json objects
        rdd = sc.textFile(','.join(files_list[files_read:index_iteration])).map(ujson.loads)

        files_read = index_iteration

        logging.info("Number of total tweets: {}".format(rdd.count()))

        t2 = datetime.now()

        logging.info("Time taken to create rdd: {}".format((t2-t1).total_seconds()))

        t3 = datetime.now()

        logging.info("Filtering out delete rows...\n")

        # remove deletes
        create_only_rdd = rdd.filter(lambda row: row.get("delete") is None)

        if create_only_rdd.isEmpty():
            logging.info("This iteration does not have any relevant tweet")
            continue

        # dates for which trading data is required

        if iteration == last_iteration:
            date_rdd = create_only_rdd.map(lambda row: row.get("created_at")).reduce(lambda a,b: max(a,b))
            max_date = utc_to_est(date_rdd).date()

        if iteration == 1:
            date_rdd = create_only_rdd.map(lambda row: row.get("created_at")).reduce(lambda a,b: min(a,b))
            min_date = utc_to_est(date_rdd).date()

        t4 = datetime.now()

        logging.info("Time taken to remove deletes: {}".format((t4-t3).total_seconds()))

        # filter create_only_rdd to get relevant tweets

        logging.info("Filtering the data...")

        t5 = datetime.now()

        filtered_rdd = create_only_rdd.filter(filter_tweet).persist()

        create_only_rdd.unpersist()

        if filtered_rdd.isEmpty():
            logging.info("This iteration does not have any relevant tweet")
            continue

        total_tweets = filtered_rdd.count()

        logging.info("Number of filtered tweets: {}".format(total_tweets))

        t6 = datetime.now()

        logging.info("Time taken to filter: {}".format((t6-t5).total_seconds()))

        logging.info("Parsing filtered tweets to extract only required fields and Performing sentiment analysis...\n")

        """ removing the extra attributes from each tweet metadata and get a rdd with only the create date,
        hour and weighted sentiment
        eg: [{'created_hour': 3, 'created_on': datetime.date(2017, 1, 26),'stock_sentiment': [0, 0, 0, 0, 0.6]}]
        sentiment is in an order of stocks and example shown says the particular tweet is related to IBM and its
        sentiment is 0.6 on the scale of -7 to 7 (based on 6 factors and 1 to avoid multiplying sentiment by 0)
        """

        t7 = datetime.now()

        tweet_with_sentiment_rdd = filtered_rdd.map(parse_tweet)

        logging.info("Numbers of tweets parsed (real tweets): {}".format(tweet_with_sentiment_rdd.count()))

        logging.info("Example of parsed tweet: {}".format(tweet_with_sentiment_rdd.first()))

        filtered_rdd.unpersist()

        t8 = datetime.now()

        logging.info("Time taken to parse tweets: {}".format((t8-t7).total_seconds()))

        logging.info("Aggregating tweets sentiment by date on which it has to be used for trading...\n")

        t9 = datetime.now()

        # get a rdd with calendar date as key and list of tuples as value
        # Each tuple represents the sum of sentiment for a stock for the day and number of tweets that provided
        # this sum of sentiments
        # With this we will get weighted sentiment for each trading day later

        date_sentiment_rdd = tweet_with_sentiment_rdd.map(lambda a: (a["created_on"],
                                                                     [(a["stock_sentiment"][0],1 if a["stock_sentiment"][0] !=0 else 0),
                                                                      (a["stock_sentiment"][1],1 if a["stock_sentiment"][1] !=0 else 0),
                                                                      (a["stock_sentiment"][2],1 if a["stock_sentiment"][2] !=0 else 0),
                                                                      (a["stock_sentiment"][3],1 if a["stock_sentiment"][3] !=0 else 0),
                                                                      (a["stock_sentiment"][4],1 if a["stock_sentiment"][4] !=0 else 0)])).\
            reduceByKey(lambda a, b: [(a[i][0] + b[i][0], a[i][1] + b[i][1]) for i in range(5)])

        logging.info("Example of tweet with sentiment: {}".format(date_sentiment_rdd.first()))

        tweet_with_sentiment_rdd.unpersist()

        t10 = datetime.now()

        logging.info("Time taken to get sentiment for date: {}".format((t10-t9).total_seconds()))

        # concatenate all the filtered rdds to one. The number of records in this rdd
        # will be less than or equal to the number of calendar days within the period
        # for which we have tweets data

        if final_date_sentiment_rdd is None:
            final_date_sentiment_rdd = date_sentiment_rdd
        else:
            final_date_sentiment_rdd.union(date_sentiment_rdd)

        final_date_sentiment_rdd.persist()

        date_sentiment_rdd.unpersist()

    # final_date_sentiment_rdd contains relevant tweets and their sentiment based on which trading can be done.
    # If it is empty then the data did not have even a single relevant tweet

    if final_date_sentiment_rdd is None:
        with open("Results.txt", 'w') as f:
            f.write("Given data had no tweet based on which trading decision could be made.\n")
            f.write("Hence sharpe ratio is zero")
        sys.exit()

    final_date_sentiment_rdd.persist()

    # By now we have sentiment analysis for all relevant tweets. Before further processing we need stock price data
    # so that we can aggregate tweets according to trading date

    logging.info("Getting stock price data...\n")

    #logging.info("Min Date: {}, Max Date: {}".format(min_date, max_date))

    t11 = datetime.now()

    stock_prices_df = sql_sc.createDataFrame(get_stocks_price_data(min_date, max_date))

    #logging.info("Preview of stock price dataframe: \n")

    logging.info("Number of trading days for which stock price is taken: {}\n".format(stock_prices_df.count()))

    t12 = datetime.now()

    logging.info("Time taken to get stock prices: {}".format((t12-t11).total_seconds()))

    # reduce sentiments to trading days only

    logging.info("Aggregating tweets sentiment by trading date on which it has to be used...\n")

    t13 = datetime.now()

    # get the trading dates column from dataframe

    date_col = copy.deepcopy(stock_prices_df.select(stock_prices_df.Date).collect())

    date_col = [date_col[i]["Date"].encode('utf-8') for i in range(len(date_col))]

    # maps date of each record to the trading date, i.e., if date is some saturday or holiday it maps
    # it to the next trading day as the sentiment will be used on that date. Then aggregates the sentiments
    # for each trading day and map them to weighted sentiment for the date

    sentiment_by_trading_day = final_date_sentiment_rdd.map(lambda (k, v): (get_trading_date(k, date_col), v)).\
        reduceByKey(lambda (a, b): [(a[i][0] + b[i][0], a[i][1] + b[i][1]) for i in range(5)]).\
        map(lambda(k, v): (k,[v[i][0]/v[i][1] if v[i][1]>0 else v[i][0] for i in range(5)]))

    logging.info("Example of tweet with sentiment by trading day: {}".format(sentiment_by_trading_day.first()))
    #logging.info("#######")
    t14 = datetime.now()

    logging.info("Time taken to get sentiment for trading date: {}".format((t14-t13).total_seconds()))

    # convert stock prices df to rdd with date as key and array of prices for that date as value

    logging.info("Converting stock price df to rdd...\n")

    t15 = datetime.now()

    stock_prices_rdd = stock_prices_df.rdd.map(lambda row: (datetime.strptime(row["Date"]
                    .encode('utf-8'),'%Y-%m-%d').date(), [row["YHOO"], row["MSFT"], row["SBUX"], row['NVDA'], row["IBM"]]))

    logging.info("Example of stock price data rdd: {}".format(stock_prices_rdd.first()))

    t16 = datetime.now()

    logging.info("Time taken to convert stock price df to rdd: {}".format((t16-t15).total_seconds()))

    # merge the sentiment and price rdds to get rdd with date as key and
    # tuple of price array and sentiment array as value

    logging.info("Merging stock prices rdd and tweet sentiment rdd by trading date...\n")

    t17 = datetime.now()

    price_sentiment_map = stock_prices_rdd.join(sentiment_by_trading_day).collectAsMap()

    logging.info("Example of stock price and sentiment dictionary: {}".format(price_sentiment_map))

    t18 = datetime.now()

    logging.info("Time taken to merge stock price and sentiment for trading date: {}".format((t18-t17).total_seconds()))

    # perform trading; this has to be processed sequentially

    initial_money_value = 1000000000
    unused_cash = 500000000
    # initial_stock_positions = [100000000, 100000000, 100000000, 100000000, 100000000]
    # order of stocks ["YHOO", "MSFT", "SBUX", 'NVDA', "IBM"]
    trading_dates = [datetime.strptime(date_col[i],'%Y-%m-%d').date() for i in range(len(date_col))]
    current_stock_positions = [100000000, 100000000, 100000000, 100000000, 100000000]
    initial_prices = stock_prices_rdd.first()[1]
    first_trading_day = trading_dates[0]
    last_trading_date = max_date
    num_stocks = [current_stock_positions[i]/initial_prices[i] for i in range(5)]
    current_money_value = 1000000000
    daily_returns = []

    # based on the trading decision calculates the worth each day and gets the daily returns

    for date in trading_dates[1:]:

        if date in price_sentiment_map.keys():
            new_stock_positions, new_unused_cash, new_num_stocks = trading(price_sentiment_map[date][1]
                                                                           , price_sentiment_map[date][0]
                                                                           , current_stock_positions, unused_cash
                                                                           , num_stocks)

            today_stock_price = stock_prices_rdd.filter(lambda x: x[0] == date).collectAsMap()[date]
            today_value = new_unused_cash + sum([new_num_stocks[i]*today_stock_price[i] for i in range(5)])

            today_return = (today_value - current_money_value)*100 / current_money_value
            daily_returns.append(today_return)

            current_stock_positions = new_stock_positions
            num_stocks = new_num_stocks
            unused_cash = new_unused_cash
            current_money_value = unused_cash + sum(current_stock_positions)

        else:
            if date <= max_date:
                daily_returns.append(0)
            else:
                #logging.info(str(date) + " does not have any sentiment")
                last_trading_date = date
                break

    t19 = datetime.now()

    logging.info("Time taken to do trading: {}".format((t19-t18).total_seconds()))

    # stocks closing price on the last trading day to get final value

    stock_price_last_date = stock_prices_rdd.filter(lambda x: x[0] == last_trading_date)\
        .collectAsMap()[last_trading_date]

    final_value = unused_cash + sum([num_stocks[i]*stock_price_last_date[i] for i in range(5)])

    # calculations for sharpe ratio (mean-5%)/std.dev and annualized sharpe ratio

    std_dev = stdev(daily_returns)
    avg_daily_return = mean(daily_returns)

    annualized_return = avg_daily_return * 365
    annualized_std_dev = std_dev * (365**(1/2))

    sharpe_ratio = (avg_daily_return - 5) / std_dev
    sharpe_ratio_annual = (annualized_return - 5) / annualized_std_dev

    with open("Results.txt", 'w') as f:

        f.write("\n######################################################\n")
        f.write("Results of the program written by Team Spark With Py\n")
        f.write("######################################################\n")

        f.write("\nSharpe ratio: {}".format(round(sharpe_ratio,4)))
        f.write("\nAnnualized Sharpe ratio: {}\n".format(round(sharpe_ratio_annual,4)))

        f.write("\nBelow are the other useful insights\n")

        f.write("\nFirst trading date: {}\n".format(first_trading_day))
        f.write("Last trading date: {}\n".format(last_trading_date))

        f.write("\nInitial value: ${}\n".format(initial_money_value))
        f.write("Final value: ${}\n".format(final_value))

        f.write("\nAverage daily returns: {}\n".format(avg_daily_return))
        f.write("Standard Deviation: {}\n".format(std_dev))

        f.write("\nAnnualized return: {}\n".format(annualized_return))
        f.write("Annualized std dev: {}\n".format(annualized_std_dev))

        # f.write("\nDaily_returns: {}\n".format(daily_returns))

    logging.info("Total Time taken: {}".format((datetime.now()-t0).total_seconds()))


"""
spark-submit --master=local[*] --num-executors 17 --executor-cores 5 --executor-memory 19G read_data_test.py TwitterData 240 --log=info

Since master is local, the rest of the spark arguments does not get consumed by spark so mentioned any number

number of command line arguments
3
command line arguments printed

['/Users/monicadabas/Desktop/MSCourseWork/Spring2017/Big_Data-STA9497/Project2/read_data_test.py', 'TwitterData', '240]

"""

# This is where the program starts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.info("Incorrect number of arguments provided")
        sys.exit()
    else:
        main(sys.argv[1], int(sys.argv[2]))


