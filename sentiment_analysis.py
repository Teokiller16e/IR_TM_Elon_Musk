import csv
import pandas as pd
import re
import os
import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import twitter_samples
from textblob import TextBlob as tb

# Download the first time
# nltk.download('twitter_samples')
# Global
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
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


def create_word_cloud(tweets_df):
    width = 1000
    height = 1000
    random_state = 21
    max_font_size = 100

    words = " ".join([tweet for tweet in tweets_df['text']])
    word_cloud = WordCloud(width=width, height=height, random_state=random_state, max_font_size=max_font_size).generate(words)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


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
# tweets_df.columns = ['Tweets']
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


# Andrea ta stopwords exoun ola ta negation mesa opote eftiaksa by hand thn default lista pou tha xrisimopoiisoume gia na gleitwnoume lekseis kai parallila na mhn exei tis
# arniseis mesa ( an deis to nltk.corpus stopwords exei couldn't,shouldn't ect και τα χρειαζόμαστε αυτά)
"""
# Remove repetitions in tokenized sentences:
sentences = word_tokenize(speeches)

duplicates = []
cleaned = []
for s in sentences:
    if s in cleaned:
        if s in duplicates:
            continue
        else:
            duplicates.append(s)
    else:
        cleaned.append(s)

# We use the cleaned list for the text that doesn't contain duplicates

# Remove stop words, with holding negations:
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "should", "now"]

"""