import spacy
from spacy.lang.en.stop_words import STOP_WORDS

"""
StopWords In Spacy
A Stop word/Stop list
Words filtered out before preprocessing 
Most Common words

Uses
Improve performance in search engines
   eg: how to perform sentiment analysis with spacy
Eliminating noise and distraction sentiment classification
   Make ML learning faster due to less features
   Make Prediction more accurate due to noise reduction
"""

nlp = spacy.load('en')
print(STOP_WORDS)
print(len(STOP_WORDS))

print(nlp.vocab["the"].is_stop)
print(nlp.vocab["theme"].is_stop)

print("*****Filtering Non Stop Words***********")
#Filtering Non Stop Words
mysentence = nlp(u"This is a sentence about how to use stopwords in natural language")

for word in mysentence:
    if word.is_stop == True:
        print(word)

print("******Filterning Non Stop Words**********")

for word in mysentence:
    if word.is_stop == False:
        print(word)


word_list = [word for word in mysentence if word.is_stop == False]

print(word_list)

#Adding Your Stop Words
STOP_WORDS.add("LoL")

print(nlp.vocab["LoL"].is_stop)
