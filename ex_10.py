import spacy
from collections import Counter

#How to find the most common words using spacy and python

nlp = spacy.load('en')
docx = nlp(open('my_spacy_file.txt').read())
print(docx)

#Remove Punc, stopwords

nouns = [token.text for token in docx if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]
print(nouns)

word_frequency = Counter(nouns)
common_nouns = word_frequency.most_common(10)

print("*******NOUNS**********")
print(common_nouns)

#Remove Punc, stopwords
print("******VERBS***********")
verbs = [token.text for token in docx if token.is_punct != True and token.pos_ == "VERB"]
print(verbs)

print("******COMMON VERBS***********")
word_frequency = Counter(verbs)
common_verbs = word_frequency.most_common(10)
print(common_nouns)

print("******COMMON VERBS with stop words***********")
verbs_with_stopwords = [token.text for token in docx if token.is_stop != True and token.is_punct != True and token.pos_ == "VERB"]

print(Counter(verbs_with_stopwords).most_common(10))