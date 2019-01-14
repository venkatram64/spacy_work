"""
NLP With SpaCy - Extending SpaCy

    Doc Document
    Tokens
    Span

Usefulness
    Allows you to add extra functionality to SpaC
        eg: sentiment analysis
    extend the API to become more accessible

Creating an Extension
    set_extension
    3 Types
        Attribute Extension
        Property Extension(getter, setter)
        Method Extension(method)

Calling the extension
"""
#C:\Users\Venkatram\AppData\Roaming\nltk_data
import spacy
from spacy.tokens import Doc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download()
nlp1 = spacy.load('en')
print(nlp1.pipeline)

sent_analyzer = SentimentIntensityAnalyzer()
def sentiment_scores(docx):
    return sent_analyzer.polarity_scores(docx.text)

Doc.set_extension("sentimenter", getter=sentiment_scores)

nlp = spacy.load('en')
ex1 = nlp("This movie was very nice.")

#Calling extenstion
print(ex1._.sentimenter)