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

# punctuation = re.compile(r'[-.?!,:;()[0-9]')


def preprocess(reply):
    reply = word_tokenize(reply)
    clean_reply = ""
    for word in reply:
        clean_reply += word + " "
    return clean_reply


replies = []
polarities = []
scores = []
subjectivities = []
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
    subjectivity = body.sentiment.subjectivity
    polarities.append(polarity)
    scores.append(score)
    subjectivities.append(subjectivity)

plt.figure()
positives = 0
neutral = 0
negatives = 0
for i in range(0, len(scores)):
    if polarities[i] > 0:
        positives += 1
    elif polarities[i] == 0:
        neutral += 1
    else:
        negatives += 1
    plt.scatter(polarities[i], scores[i], color="Blue")
plt.title("Sentiment Analysis vs Upvotes")
plt.xlabel("Polarity")
plt.ylabel("Upvotes Score")
plt.show()

fig = plt.figure()
for i in range(0, len(scores)):
    plt.scatter(polarities[i], subjectivities[i], color="Blue")
plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()

print("There are", positives, "positives")
print("There are", negatives, "negatives")
print("There are", neutral, "neutrals")