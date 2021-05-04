import csv
import pandas as pd
import re
import os
import nltk
import matplotlib.pyplot as plt
import praw
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import twitter_samples
from textblob import TextBlob as tb

stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
punctuation = re.compile(r'[-.?!,:;()[0-9]')


def preprocess(reply):
    reply = word_tokenize(reply)
    clean_reply = ""
    for word in reply:
        if word.lower() not in stopwords:
            word = lemmatizer.lemmatize(word)
            word = punctuation.sub("", word)
            clean_reply += word + " "
    return clean_reply


def sentiment_analysis(reply):
    reply = preprocess(reply)
    sentiment = tb.sentiment(reply)
    return sentiment.polarity


replies = []
polarities = []
scores = []
reddit = praw.Reddit(client_id="retmTurNtUpV4g",
                     client_secret="ytQWmu0VzUfH4DWicHIALd10EUQHkg",
                     username="irtmproject2021",
                     password="irtmproject2021",
                     user_agent="anything")

user = reddit.redditor('ElonMuskOfficial')
ids = ["590wi9", "2rgsan"]
for id in ids:
    submission = reddit.submission(id=id)
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        for reply in comment.replies:
            if reply.author == user:
                replies.append(reply)
for reply in replies:
    body = preprocess(reply.body)
    body = tb(body)
    polarity = body.sentiment.polarity
    score = reply.score
    polarities.append(polarity)
    scores.append(score)

plt.figure()
for i in range(0, len(scores)):
    plt.scatter(polarities[i], scores[i], color="Blue")
plt.title("Sentiment Analysis vs Upvotes")
plt.xlabel("Polarity")
plt.ylabel("Upvotes score")
plt.show()
