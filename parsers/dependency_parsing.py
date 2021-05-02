import numpy 
import nltk
from spacy.lang.en import English
import spacy
from spacy import displacy
import os

# Dependency parsing is the process of analyzing the grammatical structure of a sentence based on the dependencies between the words in a sentence


# 1) install anaconda (software)
# for cmd :
# 2) pip install spacy
# 3) python -m spacy download en_core_web_sm


# TODO : bug fix propably some punctuation causes that line 38
# TODO : use displacy library to achieve a better node graph

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return nltk.Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)

en_nlp = spacy.load('en_core_web_sm')
# current directory IR_TM_ELON_MUSK changed--> Datasets/text_files
cwd = os.getcwd()
os.chdir('Datasets/txt_files')
path = os.getcwd()

f = open(path+"/Elon_Musk_Dataset.txt","r")


for line in f:
    #print(line)
    test = en_nlp(line)
    print(test)
    [to_nltk_tree(sent.root).pretty_print() for sent in test.sents]


#    displacy.render(nlp(text),jupyter=True)

    #command="Submit debug logs to project lead today at 9:00 AM"
    #print("hello world")
    #en_doc=en_nlp(u''+command)

    