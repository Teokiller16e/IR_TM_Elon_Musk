from itertools import groupby
import pandas as pd
from os import read
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt 


# inserted the 2020 dataset just because the 2021 is half so it doesn't really make any difference
dataset = pd.read_csv(Path("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/dataset_2020.csv"))

dataset = dataset.assign(Time=pd.to_datetime(dataset.date)).drop('id', axis='columns') # filter by date to datetime / columns
print(dataset.head(3)) # testing if the dataset heads can be printed 

# Elon Musk tweets statistics:
sns.set_style("darkgrid")
ans = input('Specify the frequency of the tweets per -->Give  a)hour, b)day or c)month :\n')
if(ans=='hour'):
    (dataset.Time.dt.hour.value_counts().sort_index()).plot.bar(figsize=(14,7),fontsize=13,color='coral') 
    plt.gca().set_title('@Elonmusk tweets per hour of a day frequency',fontsize=20)
elif(ans=='day'):
    (dataset.Time.dt.day.value_counts().sort_index()).plot.bar(figsize=(14,7),fontsize=13,color='cyan')
    plt.gca().set_title('@Elonmusk tweets per day of a month frequency',fontsize=20)
elif(ans=='month'):
    (dataset.Time.dt.month.value_counts().sort_index()).plot.bar(figsize=(14,7),fontsize=13,color='lightgreen')
    plt.gca().set_title('@Elonmusk tweets per month of a year frequency',fontsize=20)
else:
    (dataset.Time.dt.year.value_counts().sort_index()).plot.bar(figsize=(14,7),fontsize=13,color='magenta')
    plt.gca().set_title('@Elonmusk tweets per year frequency',fontsize=20)

plt.show()

# prints the proportion of how many thumbnails/photos/videos have been posted on the tweets.
print("Thumbnails frequency :\n",dataset['thumbnail'].notnull().value_counts() / len(dataset)) 
print("\n")
print("Photo frequency :\n",dataset['video'].notnull().value_counts() / len(dataset))
print("\n")
print("Photo frequency :\n",dataset['photos'].notnull().value_counts() / len(dataset))
