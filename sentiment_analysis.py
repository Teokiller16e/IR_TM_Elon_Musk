import csv
import pandas as pd
import re
import os
import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from textblob import TextBlob as tb

# Download the first time
# nltk.download('twitter_samples')
# Global
tweets_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', 'elonmusk_tweets.csv')

# Clean The Data
def cleantext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)  # Remove Mentions
    text = re.sub(r"#", "", text)  # Remove Hashtags Symbol
    text = re.sub(r"RT[\s]+", "", text)  # Remove Retweets
    text = re.sub(r"https?:\/\/\S+", "", text)  # Remove The Hyper Link
    text = re.sub(r"\s\s+", " ", text)  # Remove Unnecessary Spaces
    text = re.sub("b[\"|\']\s*", "", text)  # Remove b"
    return text


def sentiment_analysis(tweets):
    sentiment = tb(tweets['text']).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])


def analysis(score):
    if score < 0:
        return "Negative"
    if score == 0:
        return "Neutral"
    if score > 0:
        return "Positive"


tweets_df = pd.read_csv(tweets_dataset)
tweets_df = tweets_df['text']
tweets_df = tweets_df.to_frame()
tweets_df['text'] = tweets_df['text'].apply(cleantext)
tweets_df[["Subjectivity", "Polarity"]] = tweets_df.apply(sentiment_analysis, axis=1)
# create_word_cloud(tweets_df)
tweets_df["Analysis"] = tweets_df["Polarity"].apply(analysis)
positive_tweets = tweets_df[tweets_df['Analysis'] == 'Positive']
negative_tweets = tweets_df[tweets_df['Analysis'] == 'Negative']
neutral_tweets = tweets_df[tweets_df['Analysis'] == 'Neutral']
print("There are ", len(positive_tweets), " positive tweets.")
print("There are ", len(negative_tweets), " negative tweets.")
print("There are ", len(neutral_tweets), " neutral tweets.")
plt.figure()
for i in range(0, tweets_df.shape[0]):
    plt.scatter(tweets_df["Polarity"][i], tweets_df["Subjectivity"][i], color="Blue")
plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()
