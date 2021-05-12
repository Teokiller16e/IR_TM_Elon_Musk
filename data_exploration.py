from itertools import groupby

import pylab as pyl
import pandas as pd
from os import read
import numpy as np
from pathlib import Path
# import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

# TODO: Add more details about retweets/num_of_likes etc.


# inserted the 2020 dataset just because the 2021 is half so it doesn't really make any difference
# dataset = pd.read_csv(Path("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/dataset_2020.csv"))
tweets_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', 'dataset_2020.csv')
dataset = pd.read_csv(tweets_dataset)
dataset = dataset.assign(Time=pd.to_datetime(dataset.date)).drop('id',
                                                                 axis='columns')  # filter by date to datetime / columns
#
# print(dataset['Time'][:10])
weekday = [0] * 365
week = [0] * 365
days_of_year = [0] * 365
tweets2020 = [d for d in dataset['Time'] if d.year == 2020]
for d in tweets2020:
    date = datetime.datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S")
    days_of_year[date.timetuple().tm_yday - 1] += 1
max_activity = max(days_of_year)
min_activity = min(days_of_year)

for idx, day in enumerate(days_of_year):
    week[idx] = (idx // 7) + 1
    weekday[idx] = (idx % 7) + 1
df = pd.DataFrame({"weekday": weekday, "week": week, "activity": days_of_year})
df.drop_duplicates(subset=["weekday", "week"], inplace=True)
df = df.pivot(columns="week", index="weekday", values="activity")
Weekday, Week = np.mgrid[:df.shape[0]+1, :df.shape[1]+1]
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_aspect("equal")
plt.pcolormesh(Week, Weekday, df.values, cmap="Greens", edgecolor="w", vmin=-10, vmax=100)
plt.colorbar()
plt.title("Elon Musk's Tweets in 2020")
plt.xlabel("Week of the year")
plt.ylabel("Day of the week")
ax.set_yticks(np.arange(0, 7, 2))
ax.set_xticks(np.arange(0, 53, 5))

plt.show()

# Elon Musk tweets statistics:
# sns.set_style("darkgrid")
ans = input('Specify the frequency of the tweets per -->Give  a)hour, b)day or c)month :\n')
if ans == 'hour':
    (dataset.Time.dt.hour.value_counts().sort_index()).plot.bar(figsize=(14, 7), fontsize=13, color='coral')
    plt.gca().set_title('@Elonmusk tweets per hour of a day frequency', fontsize=20)
elif ans == 'day':
    (dataset.Time.dt.day.value_counts().sort_index()).plot.bar(figsize=(14, 7), fontsize=13, color='cyan')
    plt.gca().set_title('@Elonmusk tweets per day of a month frequency', fontsize=20)
elif ans == 'month':
    (dataset.Time.dt.month.value_counts().sort_index()).plot.bar(figsize=(14, 7), fontsize=13, color='lightgreen')
    plt.gca().set_title('@Elonmusk tweets per month of a year frequency', fontsize=20)
else:
    (dataset.Time.dt.year.value_counts().sort_index()).plot.bar(figsize=(14, 7), fontsize=13, color='magenta')
    plt.gca().set_title('@Elonmusk tweets per year frequency', fontsize=20)

plt.show()

# prints the proportion of how many thumbnails/photos/videos have been posted on the tweets.
print("Thumbnails frequency :\n", dataset['thumbnail'].notnull().value_counts() / len(dataset))
print("\n")
print("Video frequency :\n", dataset['video'].notnull().value_counts() / len(dataset))
print("\n")
print("Photo frequency :\n", dataset['photos'].notnull().value_counts() / len(dataset))

# Top retweet resources of Elon Musk:
# Loading different file because the other does not contain resources from retweets
# dataset = pd.read_csv(Path("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/data_elonmusk.csv"))
tweets_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', 'data_elonmusk.csv')
dataset = pd.read_csv(tweets_dataset)
dataset = dataset.assign(Time=pd.to_datetime(dataset.Time)).drop('row ID',
                                                                 axis='columns')  # filter by date to datetime / columns
# print(dataset.head(3))

# get head(100) top 100 tweeter sources of Elon Musk
dataset['Retweet from'].value_counts().head(10).plot.bar(figsize=(21, 11), fontsize=16, color='yellow')
plt.gca().set_title('@Elonmusk top retweet sources  (first 10 retweets) ', fontsize=20)
plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right', fontsize=14)
plt.show()
