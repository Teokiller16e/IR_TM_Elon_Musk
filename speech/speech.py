import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Need to run this the first time
# nltk.download('punkt')
# nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

filename = os.path.join(os.path.pardir, 'Datasets', 'txt_files', 'Elon_Musk_Dataset.txt')
with open(filename, "r") as file:
    speeches = "".join(file.readlines()[0:])
lemmatized_speeches = []
speeches = re.sub(r'[^\w\s]', '', speeches)  # cleaning
words = nltk.word_tokenize(speeches)  # tokenization
words = [w for w in words if not w in stop_words]  # stopwords removal
for word in words:
    if word not in stop_words:
        lemmatized_speeches.append(lemmatizer.lemmatize(word))  # lemmatizing
print(lemmatized_speeches)
