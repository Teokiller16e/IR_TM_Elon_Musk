import re
import numpy as np
import os
import pandas as pd
import xlsxwriter
from textblob import TextBlob as tb
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import cdist

filename = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', '2012.csv')
file_andreas = os.path.join(os.path.curdir, 'Rates', 'andreas_rates.txt')
file_teo = os.path.join(os.path.curdir, 'Rates', 'teo_rates.txt')
tweets_df = pd.read_csv(filename)
tweets_df = tweets_df['tweet']
out_workbook = xlsxwriter.Workbook("sentiment.xlsx")
out_sheet = out_workbook.add_worksheet('Sheet1')
tweets100 = tweets_df.head(100).tolist()
rates = []
tbrates = []

def give_rates():
    output = os.path.join(os.path.curdir, 'rates.txt')
    out = open(output, 'w')
    for id, tweet in enumerate(tweets_df.head(100)):
        print(tweet)
        rate = input("Rate the polarity of the tweet:  ")
        temp = "" + str(id) + ".: " + str(rate) + "\n"
        out.write(temp)


def get_mean_rates():
    teo_rates = []
    andreas_rates = []
    mean_rates = []
    with open(file_teo, 'r') as file_t:
        for line in file_t:
            temp_rate = line.split(':')[1]
            temp_rate = temp_rate.split('\n')[0]
            temp_rate = float(temp_rate)
            teo_rates.append(temp_rate)
    with open(file_andreas, 'r') as file_a:
        for line in file_a:
            temp_rate = line.split(':')[1]
            temp_rate = temp_rate.split('\n')[0]
            temp_rate = float(temp_rate)
            andreas_rates.append(temp_rate)

    for i in range(len(teo_rates)):
        temp_rate = andreas_rates[i] + teo_rates[i]
        mean_rates.append(temp_rate/2)

    return mean_rates

def write_in_csv(mean_rates, tweets):
    out_sheet.write('A1', 'Tweets')
    for i, tweet in enumerate(tweets.head(100)):
        row = i + 2
        out_sheet.write(f'A{row}', tweet)
    out_sheet.write('B1', 'Polarity')
    for i, rate in enumerate(mean_rates):
        row = i + 2
        out_sheet.write(f'B{row}', rate)


def analysis(score):
    if score < 0:
        return "Negative"
    if score == 0:
        return "Neutral"
    if score > 0:
        return "Positive"

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def find_kappa_distance(mean_rates, tweets_df):
    tbrates = []
    for i, tweet in enumerate(tweets_df.head(100)):
        tbrate = tb(tweet).sentiment.polarity
        tbrates.append(tbrate)


    tbrates = [round(rate) for rate in tbrates]
    mean_rates = [round(rate) for rate in mean_rates]
    distance = euclidean(tbrates, mean_rates) / 100
    k_score = cohen_kappa_score(tbrates, mean_rates)
    print(k_score)
    print(distance)







mean_rates = get_mean_rates()
write_in_csv(mean_rates, tweets_df)
out_workbook.close()
find_kappa_distance(mean_rates, tweets_df)