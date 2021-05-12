import re
import os
import pandas as pd

filename = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', '2012.csv')
file_andreas = os.path.join(os.path.curdir, 'Rates', 'andreas_rates.txt')
file_teo = os.path.join(os.path.curdir, 'Rates', 'teo_rates.txt')
tweets_df = pd.read_csv(filename)
tweets_df = tweets_df['tweet']

tweets100 = tweets_df.head(100).tolist()
rates = []

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


mean_rates = get_mean_rates()
print(mean_rates)