from os.path import curdir
import numpy 
import nltk
from spacy.lang.en import English
import spacy
from spacy import displacy
import os
import xlsxwriter
import pandas as pd

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



outWorkbook = xlsxwriter.Workbook("depend.xlsx")
outSheet = outWorkbook.add_worksheet('Sheet1')

en_nlp = spacy.load('en_core_web_sm')
# current directory IR_TM_ELON_MUSK changed--> Datasets/text_files
cwd = os.getcwd()
os.chdir('Datasets/txt_files')
path = os.getcwd()

f_tree = open(path+"/Elon_Musk_Dataset.txt","r")
f = open(path+"/Elon_Musk_Dataset.txt").read()
f_doc = en_nlp(f)

# Building the xlsx file of adjectival identifier tags
text_list = []
dep_list = []
head_list = []

for token in f_doc:
    text_list.append(token.text)

for token in f_doc:
    dep_list.append(token.dep_)

for token in f_doc:
    head_list.append(token.head.text)


outSheet.write('A1', 'Actual_Text')
for row in range(2,len(text_list)+2):
    outSheet.write(f'A{row}', text_list[row-2])


outSheet.write('B1', 'Dependencies')
for drow in range(2,len(dep_list)+2):
    outSheet.write(f'B{drow}', dep_list[drow-2])



outSheet.write('C1', 'Heads')
for hrow in range(2,len(head_list)+2):
    outSheet.write(f'C{hrow}', head_list[hrow-2])




outWorkbook.close()


cnt = 0
for line in f_tree:
    #print(line)
    test = en_nlp(line)
    print(test)
    [to_nltk_tree(sent.root).pretty_print() for sent in test.sents]
    

