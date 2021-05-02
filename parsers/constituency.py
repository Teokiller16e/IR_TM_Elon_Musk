import benepar
import spacy
import nltk
nltk.download()
from benepar import BeneparComponent


# TODO: Find a way to print the constituency full parsing graph:

#benepar.download('benepar_en2')
benepar.download('benepar_en')
nlp = spacy.load('en_core_web_sm')

nlp.add_pipe(BeneparComponent('benepar_en'))

text="It took me more than two hours to translate a few pages of English"

a = list(nlp(text).sents)[0]._.parse_string

print(a)