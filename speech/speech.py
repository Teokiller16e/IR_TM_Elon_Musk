import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from os import read
import os
import re
from nltk.corpus.reader import tagged
from nltk.tokenize import  word_tokenize , sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords,state_union
from nltk.stem import PorterStemmer,WordNetLemmatizer
import nltk 
import sklearn


# Need to run this the first time
# nltk.download('punkt')
# nltk.download('wordnet')


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

            #Chunk 1 : Adjective followed by a Noun 

            chunkGram1 = r"""chunk: {<JJ>+<NN>+}"""

            #Chunk 2 : Noun and Adjective with other POS in between
            chunkGram2 = r"""chunk: {<NN|NNP|NNS|NNPS>+<IN|DT|NN|VB.|RB>*<JJ>+}"""

            #Chunk 3: Sequence of Nouns
            chunkGram3 = r"""chunk: {<NN|NNP|NNS|NNPS>{2,9}}"""

            chunkGram = r"""VP:{<VBD>?<TO><VB>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()

        # Entity Name Recognition
            #name_entity = nltk.ne_chunk(tagged)
            #name_entity.draw()

    except Exception as e:
        print(str(e))



stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

cwd = os.getcwd()
os.chdir('Datasets/txt_files')
path = os.getcwd()

# Specify the dataset and apply preprocessing
speeches = open(path+"/Elon_Musk_Dataset.txt",encoding='utf-8').read()
lemmatized_speeches = []
#speeches = re.sub(r'[^\w\s]', '', speeches)  # cleaning
sentences = nltk.sent_tokenize(speeches)  # tokenization
words = [w for w in sentences if not w in stop_words]  # stopwords removal
for word in words:
    if word not in stop_words:
        lemmatized_speeches.append(lemmatizer.lemmatize(word))  # lemmatizing
print(lemmatized_speeches)
lemmatized_speeches = str(lemmatized_speeches)



checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(lemmatized_speeches)
#checked_unsupervised_tokenizer = custom_sent_tokenizer.tokenize(sample_text)
process_content(checked_unsupervised_tokenizer) # call part of speech tagging function