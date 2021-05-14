from io import StringIO
import matplotlib
import numpy as np 
import os 
from nltk.tokenize import word_tokenize, sent_tokenize
import emoji # we will continue that with tweets and possible hashtags and emoji
import re
from collections import Counter
import matplotlib.pyplot as plt

cwd = os.getcwd()
os.chdir('Datasets/txt_files')
path = os.getcwd()

speeches = open(path+"/Elon_Musk_Dataset.txt",encoding='utf-8').read()
speeches = speeches.lower() # lowercase convertion

# Remove special characters 
speeches = speeches.replace(r"(http|@)\S+", "")
#speeches = speeches.apply(demojize)
speeches = speeches.replace(r"::", ": :")
speeches = speeches.replace(r"’", "'")
speeches = speeches.replace(r"[^a-z\':_]", " ")

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
final_speeches = []

for word in sentences:
    if word not in stopwords:
        final_speeches.append(word)

# final_speeches is the list of the final cleaned words

# NLP Emotion Algorithm 
emotion_list = []

with open(path+"/Emotions/emotions.txt","r") as emotions:
    for line in emotions:
        #print("Line : ",line)
        clear_line = line.replace('\n','').replace(',' ,'').replace("'", '').strip()
        word, emotion = clear_line.split(':') # separate the word until : to the word and after the : sends to the emotion
        if word in final_speeches:
            emotion_list.append(emotion)

w = Counter(emotion_list)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(),w.values(),color=['black', 'red', 'green', 'blue', 'cyan','yellow','green','coral','magenta','lightblue','lightgreen'])

fig.autofmt_xdate()
plt.title('Emotion Analysis')
plt.show()