from nltk import chunk
import numpy
import nltk
import pprint
import re
nltk.download("conll2000")
from nltk.corpus import conll2000

# TODO: implement crf/lstm custom parsers to compare with already existing

# chunking is between POS and Full language parsing (basically shallow parsing)
def preprocess (doc):
    sentences = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

if __name__ == "__main__":
    text = " The blogger taught the reader to chunk"
    sentence = preprocess(text)

# Multiple line regular expression
    #grammar =  grammar = r"""
    #    NP: {<DT>?<JJ>*<NN>} # noun phrase
    #    VP: {<VBD>?<TO>?<VB>?}     # verb phrase
	#"""
    
    grammar2 = r""
    NPChunker = nltk.RegexpParser(grammar2)
    test_sentences = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
    print(NPChunker.evaluate(test_sentences))

#    result = NPChunker.parse(sentence[0])
#    result.draw()

    # chinking is a sequence of tokens that should be excluded from a chunk and is another tool in shallow parsing.
    # NP:}<IN|DT>+{ # chinking pattern that excludes from noun phrases prepositions & determiners

    # Evaluation of our parser:
