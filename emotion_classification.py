from typing_extensions import final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdflib.graph import Seq
from scipy.sparse.construct import random
import seaborn as sns
import neattext.functions as nfx
from collections import Counter
import os
import glob
from sklearn.linear_model import LogisticRegression
from transformers import BertModel, BertConfig
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import precision_recall_fscore_support,cohen_kappa_score



def display_evaluation(y_test,predictions):
    y_true = y_test
    y_pred = predictions

    score = precision_recall_fscore_support(y_true, y_pred)
    score = np.array(score)[:3]
    cr = classification_report(y_true, y_pred)
    heatmap = sns.heatmap(data=score, vmin=0, vmax=1, xticklabels=['anger', 'fear', 'joy','neutral','sadness'],
                          yticklabels=['precision', 'recall', 'f1-score'])
    print(cr)
    kappa_distance = cohen_kappa_score(y_true, y_pred)
    print('Kappa distance:', kappa_distance)
    plt.title('Precision Recall F1-score for emotion Analysis')
    plt.show()



def mannualy_anotation(load_dataset_path):
    manually_anotated = []
    texts = []
    # Specify the dataset and apply preprocessing
    with open(load_dataset_path+"/emotions.txt",'r')as file:
        
        for line in file:
            tweet = line.split(':')[0]
            temp_emo = line.split(':')[-1]
            print(tweet)
            print(temp_emo)
            temp_emo = temp_emo.replace(' ','')
            manually_anotated.append(temp_emo.replace('\n',''))
            texts.append(tweet)
    return manually_anotated,texts



def load_dataset(load_dataset_path):

    # Specify the dataset and apply preprocessing
    speeches = open(load_dataset_path+"/Elon_Musk_Dataset.txt",encoding='utf-8').read()
    #speeches = speeches.lower() # lowercase convertion

    # Remove special characters 
    speeches = speeches.replace(r"(http|@)\S+", "")
    #speeches = speeches.apply(demojize)
    speeches = speeches.replace(r"::", ": :")
    speeches = speeches.replace(r"’", "'")
    speeches = speeches.replace(r"[^a-z\':_]", " ")

    # Remove repetitions in tokenized sentences:
    sentences = sent_tokenize(speeches)


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

    # Remove stop words, with holding negations (not,nor, etc.):

    final_speeches = []

    for word in sentences:
        final_speeches.append(word)

    return final_speeches


# Resouce Link : https://www.youtube.com/watch?v=t1TkAcSDsI8 

# Combined 2 or more different datasets to 1 
"""
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "last_version_dataset.csv", index=False, encoding='utf-8-sig')
"""


# Load training dataset 
cwd = os.getcwd()
load_dataset_path = (cwd+"/Datasets/txt_files")
os.chdir('Datasets/txt_files/Emotions')


df = pd.read_csv("last_version_dataset.csv")
print(df.shape)
sns.countplot(x='Emotion',data=df)
plt.show()
#text_field = "".join(df['Text'].astype(str))


# Text cleaning and preprocessing :'

df['Clean_Text'] = df['Text'].apply(nfx.remove_punctuations)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)

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

df['Clean_Text'] = df['Clean_Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

print(df.shape[0])
df = df.drop_duplicates()
# Machine Learning Classification
# Naive Bayes/Logistic Regression/KNN/Decision Tree
training_data = df['Clean_Text'] # remove duplicates
yLabels = df['Emotion']

counter_vect = CountVectorizer()
xData = counter_vect.fit_transform(training_data)

x_train,x_test,y_train,y_test = train_test_split(xData,yLabels, test_size=0.35, random_state=42) #random state indicates the suffling of the data

print("Select the model you want to use for emotion mining \n")
print("1-> Bayes Model\n")
print("2-> Logistic Regression model \n")
print("3-> Support vector machine model \n")


answer = int(input("Answer :\n"))
# Choose model architecture
if answer == 1:
    model = MultinomialNB()
    model.fit(x_train,y_train)
    print("Model Accuracy : %.2f%%"%(model.score(x_test,y_test)*100))
elif(answer == 2):
    model = LogisticRegression()
    model.fit(x_train,y_train)
    print("Model Accuracy : %.2f%%"%(model.score(x_test,y_test)*100))
elif(answer == 3):
    model = svm.SVC()
    model.fit(x_train,y_train)
    print("Model Accuracy : %.2f%%"%(model.score(x_test,y_test)*100))


 # Predictions of emotions:
predictions = model.predict(x_test)
#sample_text = ["I really love playing voleyball with my friends at the beach"] # Here we can open the dataset to a different dataframe and loop through tokenized sentences so we can predict all of them and hold a counter for the probabilities

emos,texts = mannualy_anotation(load_dataset_path)

#speeches = load_dataset(load_dataset_path)

models_emotions = []

for sample_text in texts:
    vect_test = counter_vect.transform([sample_text]).toarray() # if we don't convert to an numpy array it will not receive the input 
    final_pred = model.predict(vect_test)
    models_emotions.append(final_pred[0])
    print("Sentence: \n",sample_text)
    #final_pred = int(final_pred)
    print("Final prediction : ",final_pred)


display_evaluation(emos,models_emotions)


# Make actual prediction :
# Prediction accuracy percentage : 
if(answer==1 or answer ==2):
    classes = model.classes_
    classes = np.reshape(classes,(5,1))
    results = model.predict_proba(vect_test)
    results = np.reshape(results,(5,1))
    for i,val in enumerate(results):
        print(classes[i]," probability : %.2f%%"%(val*100))

# Precision, Recall & F1 score summarization
print(classification_report(y_test,predictions))
display_evaluation(y_test,predictions)
plot_confusion_matrix(model,x_test,y_test)
plt.title('Confusion Matrix')
plt.show()

