from os import read
import os
import re
from nltk.corpus.reader import tagged
from nltk.tokenize import  word_tokenize , sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords,state_union
from nltk.stem import PorterStemmer,WordNetLemmatizer
import nltk 
import sklearn


# PunktSentenceTokenizer is an unsupervised machine learning model that we can either train or use the pretrained one
with open("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/Elon_Musk_Dataset.txt","rt") as file:
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")

    speeches = "".join(file.readlines()[0:])
    #print(speeches)

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    #custom_speeches_tokenizer = PunktSentenceTokenizer(speeches)
    
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
            #name_entity = nltk.ne_chunk(tagged)
            #name_entity.draw()

    except Exception as e:
        print(str(e))


# Vesion of nltk and sklearn
print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# Reading csv file from Datasets

stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

with open("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/elonmusk_tweets.csv", "r",encoding="utf-8") as input:
    
    #reader_csv = csv.reader(input)
    reader_csv = "".join(input.readlines()[0:])
    converted_row = re.sub("[^A-Za-z0-9:/-]+"," ", str(reader_csv))

    # Tokenization for words or sentences respectively:
    tokenized_words = word_tokenize(converted_row)
    tokenized_sentences = sent_tokenize(converted_row)

    # Stopwords
    filtered_words = [w for w in tokenized_words if not w in stopwords]
    #print(filtered_words)

    lemmatized_speeches = []
    for word in filtered_words:
        if word not in stopwords:
            lemmatized_speeches.append(lemmatizer.lemmatize(word))  # lemmatizing
    #print(lemmatized_speeches)


# Stemming:
#    for word in lemmatized_speeches:
#        print(ps.stem(word))

    #checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(converted_row)
    checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(sample_text)
    process_content(checked_unsupervised_tokenizer) # call part of speech tagging function

# Reading case of 
#dataset = pd.read_csv("F:/Downloads/Practice_Projects/Natural_Language_Processing/IR_TM_Elon_Musk/Datasets/txt_files/tweets/elonmusk_tweets.csv", "rt", error_bad_lines=False)
#print(dataset.head(5))