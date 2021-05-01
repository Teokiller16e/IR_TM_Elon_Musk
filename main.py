import praw
import os
import re
import nltk
import sklearn
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus.reader import tagged
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.util import bigrams, trigrams, ngrams

# Need to run this the first time
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('state_union')
# nltk.download('averaged_perceptron_tagger')

# GLOBAL VARIABLES
speeches_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'Elon_Musk_Dataset.txt')
tweets_dataset = os.path.join(os.path.curdir, 'Datasets', 'txt_files', 'tweets', 'elonmusk_tweets.csv')
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
freddit = FreqDist()
fspeeches = FreqDist()
ftweets = FreqDist()
punctuation = re.compile(r'[-.?!,:;()[0-9]')
url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
urls = []

# PunktSentenceTokenizer is an unsupervised machine learning model that we can either train or use the pretrained one
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# urls

# Vesion of nltk and sklearn
# print('The nltk version is {}.'.format(nltk.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

def create_word_cloud(data):
    # TODO Play With Parameters
    width = 100
    height = 100
    random_state = 21
    max_font_size = 100

    words = " ".join([word for word in data])
    word_cloud = WordCloud(width=width, height=height, random_state=random_state, max_font_size=max_font_size).generate(words)
    return word_cloud


# POS Function :
def process_content(tokenized):
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

            chunkGram = r"""Chunk:{<JJ.?>*}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()

        # Entity Name Recognition
        # name_entity = nltk.ne_chunk(tagged)
        # name_entity.draw()

    except Exception as e:
        print(str(e))


def preprocess(dataset):
    dataset_clean = []
    pos = []
    for word in dataset:
        if word.lower() not in stopwords:
            word = lemmatizer.lemmatize(word)
            dataset_clean.append(punctuation.sub("", word))
            pos.append(pos_tag([word]))
    return dataset_clean, pos


# REDDIT
def get_reddit_replies():
    elons_reddit_replies = ""

    reddit = praw.Reddit(client_id="retmTurNtUpV4g",
                         client_secret="ytQWmu0VzUfH4DWicHIALd10EUQHkg",
                         username="irtmproject2021",
                         password="irtmproject2021",
                         user_agent="anything")

    user = reddit.redditor('ElonMuskOfficial')
    # First post
    submission1 = reddit.submission(id="590wi9")
    submission1.comments.replace_more(limit=0)
    for comment in submission1.comments:
        for reply in comment.replies:
            if reply.author == user:
                elons_reddit_replies += reply.body.replace('\n', ' ') + ' '

    # Second post
    submission2 = reddit.submission(id="2rgsan")
    submission2.comments.replace_more(limit=0)
    for comment in submission2.comments:
        for reply in comment.replies:
            if reply.author == user:
                elons_reddit_replies += reply.body.replace('\n', ' ') + ' '
    return elons_reddit_replies


elons_reddit_replies = get_reddit_replies()
reddit_tokennized = word_tokenize(elons_reddit_replies)
reddit_cleaned, reddit_pos = preprocess(reddit_tokennized)
for token in reddit_cleaned:
    freddit[token.lower()] += 1
print(freddit.most_common(20))
urls += re.findall(url, elons_reddit_replies)
# print(reddit_pos)
# End of Reddit


# SPEECHES
with open(speeches_dataset, "r") as file:
    speeches = "".join(file.readlines()[0:])
speeches_tokenized = word_tokenize(speeches)
speeches_cleaned, speeches_pos = preprocess(speeches_tokenized)
for token in speeches_cleaned:
    fspeeches[token.lower()] += 1
print(fspeeches.most_common(20))
urls += re.findall(url, speeches)
# print(speeches_pos)
# End of Speeches


# TWEETS

# Reading csv file from Datasets
with open(tweets_dataset, "r", encoding="utf-8") as file:
    # reader_csv = csv.reader(input)
    reader_csv = "".join(file.readlines()[0:])
    converted_row = re.sub("[^A-Za-z0-9:/-]+", " ", str(reader_csv))
    # Tokenization for words or sentences respectively:
    tweets_tokenized = word_tokenize(converted_row)
    tweets_cleaned, tweets_pos = preprocess(tweets_tokenized)
    for token in tweets_cleaned:
        ftweets[token.lower()] += 1
    print(ftweets.most_common(20))
    urls += re.findall(url, str(reader_csv))
    # print(tweets_pos)
    print(urls)
    tokenized_sentences = sent_tokenize(converted_row)
    # checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(converted_row)
    # checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(sample_text)
    # process_content(checked_unsupervised_tokenizer)  # call part of speech tagging function

# Reading case of
# dataset = pd.read_csv("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/elonmusk_tweets.csv", "rt", error_bad_lines=False)
# print(dataset.head(5))


# Apply Word Cloud
fig = plt.figure(figsize=(30, 30), dpi=20)
reddit_word_cloud = create_word_cloud(reddit_tokennized)
speeches_word_cloud = create_word_cloud(speeches_tokenized)
tweets_word_cloud = create_word_cloud(tweets_tokenized)

fig.add_subplot(1, 3, 1)
plt.imshow(reddit_word_cloud, interpolation="bilinear")
plt.title("Reddit")
plt.axis("off")
fig.add_subplot(1, 3, 2)
plt.imshow(speeches_word_cloud, interpolation="bilinear")
plt.title("Speeches")
plt.axis("off")
fig.add_subplot(1, 3, 3)
plt.imshow(tweets_word_cloud, interpolation="bilinear")
plt.title("Tweets")
plt.axis("off")
plt.show()
