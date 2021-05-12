import re
import os
import pandas as pd

filename = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', '2012.csv')

tweets_df = pd.read_csv(filename)
tweets_df = tweets_df['tweet']

tweets100 = tweets_df.head(100).tolist()
rates = []
output = os.path.join(os.path.curdir, 'rates.txt')
out = open(output, 'w')
for id, tweet in enumerate(tweets_df.head(100)):
    print(tweet)
    rate = input("Rate the polarity of the tweet:  ")
    temp = "" + str(id) + ".: " + str(rate) + "\n"
    out.write(temp)

