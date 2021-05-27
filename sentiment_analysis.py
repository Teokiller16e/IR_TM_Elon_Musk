import csv
import pandas as pd
import re
import os
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from wordcloud import WordCloud
from textblob import TextBlob as tb
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from yellowbrick.classifier import classification_report as cr

# Download the first time
# nltk.download('twitter_samples')
# Global
tweets_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', 'elonmusk_tweets.csv')
sentimentxlsx = os.path.join(os.path.curdir, 'sentiment.xlsx')


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
    sentiment = tb(tweets['Tweets']).sentiment
    # sentiment = tb(tweets['text']).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])


def analysis(score):
    if score < 0:
        return "Negative"
    if score == 0:
        return "Neutral"
    if score > 0:
        return "Positive"


def apply_sent():
    tweets_df = pd.read_csv(tweets_dataset)
    tweets_df = tweets_df['text']
    tweets_df = tweets_df.to_frame()
    tweets_df['text'] = tweets_df['text'].apply(cleantext)
    tweets_df[["Subjectivity", "Polarity"]] = tweets_df.apply(sentiment_analysis, axis=1)
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


def evaluate_sent():
    df = pd.read_excel(sentimentxlsx, engine='openpyxl')
    df['Tweets'] = df['Tweets'].apply(cleantext)
    # sentiment = tb(df['Tweets']).sentiment
    df[['tbSubjectivity', 'tbPolarity']] = df.apply(sentiment_analysis, axis=1)
    df['Analysis'] = df['Polarity'].apply(analysis)
    df['tbAnalysis'] = df['tbPolarity'].apply(analysis)

    positive_tweets = df[df['Analysis'] == 'Positive']
    negative_tweets = df[df['Analysis'] == 'Negative']
    neutral_tweets = df[df['Analysis'] == 'Neutral']
    print('In multi-juror sentiment analysis there are:')
    print("There are ", len(positive_tweets), " positive tweets.")
    print("There are ", len(negative_tweets), " negative tweets.")
    print("There are ", len(neutral_tweets), " neutral tweets.")
    true_perc = [len(positive_tweets) / 100, len(negative_tweets) / 100, len(neutral_tweets) / 100]
    positive_tweets = df[df['tbAnalysis'] == 'Positive']
    negative_tweets = df[df['tbAnalysis'] == 'Negative']
    neutral_tweets = df[df['tbAnalysis'] == 'Neutral']
    print('In textblob sentiment analysis there are:')
    print("There are ", len(positive_tweets), " positive tweets.")
    print("There are ", len(negative_tweets), " negative tweets.")
    print("There are ", len(neutral_tweets), " neutral tweets.")
    pred_perc = [len(positive_tweets) / 100, len(negative_tweets) / 100, len(neutral_tweets) / 100]
    df['equal'] = np.where(df['Analysis'] == df['tbAnalysis'], True, False)
    return df, true_perc, pred_perc


def display_evaluation(df):
    y_true = df['Analysis'].values
    y_pred = df['tbAnalysis'].values

    score = precision_recall_fscore_support(y_true, y_pred)
    score = np.array(score)[:3]
    cr = classification_report(y_true, y_pred)
    heatmap = sns.heatmap(data=score, vmin=0, vmax=1, xticklabels=['Negative', 'Neutral', 'Positive'],
                          yticklabels=['precision', 'recall', 'f1-score'])
    print(cr)
    kappa_distance = cohen_kappa_score(y_true, y_pred)
    print('Kappa distance:', kappa_distance)
    plt.title('Precision Recall F1-score for sentiment Analysis')
    plt.show()


def display_percentages(list1, list2):
    fig = plt.figure()
    subplot = fig.add_subplot()
    labels = ['Positive', 'Negative', 'Neutral']
    index = 1
    width = 0.2
    for idx in range(len(list1)):
        plt.bar(idx+1, list1[idx], width, label='Manually Annotated')
        plt.bar(idx+1+width, list2[idx], width, label='Predicted')

    plt.legend()
    plt.xticks([1, 2, 3], labels)
    plt.title('Percentages of each class')
    plt.show()


df, perc_true, perc_pred = evaluate_sent()
display_evaluation(df)
display_percentages(perc_true, perc_pred)
