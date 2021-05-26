import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
import neattext.functions as nfx
from collections import Counter
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize


# Resouce Link : https://www.youtube.com/watch?v=t1TkAcSDsI8 

# Combined 2 different datasets to 1 
"""
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "merged_emotions.csv", index=False, encoding='utf-8-sig')

"""

# Load training dataset 
cwd = os.getcwd()
os.chdir('Datasets/txt_files/Emotions')



df = pd.read_csv("merged_emotions.csv")
print(df.shape)
sns.countplot(x='Emotion',data=df)
plt.show()
#text_field = "".join(df['Text'].astype(str))


# Text cleaning and preprocessing :'

df['Clean_Text'] = df['Text'].apply(nfx.remove_punctuations)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)
stopwords = ["I", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "should", "now"]
final_text = []

df['Clean_Text'] = df['Clean_Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))



print(df.shape[0])
df = df.drop_duplicates()
# Machine Learning Classification
# Naive Bayes/Logistic Regression/KNN/Decision Tree
training_data = df['Clean_Text'] # remove duplicates
yLabels = df['Emotion']

counter_vect = CountVectorizer()
xData = counter_vect.fit_transform(training_data)

x_train,x_test,y_train,y_test = train_test_split(xData,yLabels, test_size=0.2, random_state=42) #random state indicates the suffling of the data

# Build machine learning model:
bayes_model = MultinomialNB()
bayes_model.fit(x_train,y_train)
bayes_model.score(x_test,y_test)

# Predictions of emotions:
predictions = bayes_model.predict(x_test)

sample_text = ["I really love playing voleyball with my friends at the beach"] # Here we can open the dataset to a different dataframe and loop through tokenized sentences so we can predict all of them and hold a counter for the probabilities
vect_test = counter_vect.transform(sample_text).toarray() # if we don't convert to an numpy array it will not receive the input 

# Make actual prediction :
final_pred = bayes_model.predict(vect_test)
print("Final prediction : ",final_pred)

# Prediction accuracy percentage : 
classes = bayes_model.classes_
classes = np.reshape(classes,(5,1))

results = bayes_model.predict_proba(vect_test)
results = np.reshape(results,(5,1))
for i,val in enumerate(results):
    print(classes[i]," probability : %.2f%%"%(val*100))

# Precision, Recall & F1 score summarization



# Compare SparkNLP / NLU John Snows Lab
# Load the packages:


