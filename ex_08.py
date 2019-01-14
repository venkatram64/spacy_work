import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import string
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


"""
Text Classification With Machine Learning and SpaCy
Text categorization / text classification is the task of assigning predefined categories to document
Sentiment Analysis
Multilabel classification
    http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
"""

df_yelp = pd.read_table('yelp_labelled.txt')
df_imdb = pd.read_table('imdb_labelled.txt')
df_amazon = pd.read_table('amazon_cells_labelled.txt')

my_freames = [df_yelp, df_imdb, df_amazon]

#Renaming Column Headers
for colname in my_freames:
    colname.columns = ["Message", "Target"]

#Column names
for colname in my_freames:
    print(colname.columns)

#Assign a key to make it easier
keys = ["Yelp", "IMDB", "Amazon"]

#Merge or Concat our datasets
df = pd.concat(my_freames, keys=keys)

#length and shape
print(df.shape)

print(df.head())

df.to_csv("sentiment_dataset.csv")
#Data Cleaning
print(df.columns)

#checking for missing values
print(df.isnull().sum())

df_clean = df

"""
Working with SpaCy
Removing Stopwords
Lemmatizing
"""

nlp = spacy.load('en')

#Build a list of stopwords to use to filter
stopwords = list(STOP_WORDS)

print(stopwords)

#Getting Lemma and Stop words

docx = nlp("This is how John walker was walking. He was also running beside the lawn")

#Lemmatizing of tokens
for word in docx:
    print(word.text, " Lemma : ", word.lemma_)

#Lemma that are not pronouns
for word in docx:
    if word.lemma_ != '-PRON-':
        print(word.lemma_.lower().strip())

#List Comprehensions of our Lemma

#[word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in docx]

#Flitering out Stopwords and Punctuations
for word in docx:
    if word.is_stop == False and not word.is_punct:
        print(word)

#Stop words and Punctuation in List Comprehension
#[word for word in docx if word.is_stop == False and not word.is_punct]

mysents = []
for i in df_clean.Message:
    docx = nlp(i)
    procs = [word for word in docx if word.is_stop == False and not word.is_punct]
    mysents.append(procs)

#Use the punctuations of string module
punctuations = string.punctuation

#Creating a Spacy Parser

parser = English()

def spacy_tokenizer(sentence):
    my_tokens = parser(sentence)
    my_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in my_tokens]
    my_tokens = [word for word in my_tokens if word not in stopwords and word not in punctuations]

    return my_tokens


#Machine Learning with sklearn

#Custom transformer using SpaCy

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    #Basic function to clean the text
    def clean_text(self, text):
        return text.strip().lower()



vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))

classifier = LinearSVC()

#Using Tfidf
tfvectorizor = TfidfVectorizer(tokenizer=spacy_tokenizer)

#Features and Labels
X = df['Message']
ylabels = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.20, random_state=42)

#Create the pipeline to clean, tokenize, vectorize and classify

pipe = Pipeline([("cleanner", predictors()),
                 ("vectorizer", vectorizer),
                 ('classifier', classifier)])

#Fit our data

pipe.fit(X_train, y_train)

#Predicting with a test dataset

sample_prediction = pipe.predict(X_test)

#Prediction Results
# 1 = Positive review
# 0 = Negative review

for (sample, pred) in zip(X_test, sample_prediction):
    print(sample, " Prediction => ", pred)

#Accuracy

print("Accuracy: ", pipe.score(X_test, y_test))
print("Accuracy: ", pipe.score(X_test, sample_prediction))

print("Accuracy: ", pipe.score(X_train, y_train))

#Another random review

print(pipe.predict(["This was great movie"]))

example=["I do enjoy my job",
         "What a poor product!, I will have to get a new one",
         "I feel amazing!"]

print(pipe.predict(example))




