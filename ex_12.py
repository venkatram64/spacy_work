"""
TextaCy
TextaCy is a Python Library for performing higher leverl natural language processing(NLP) tasks
built on the high performance SpaCy library
TextaCy focuses on tasks facilitated by the availability of tokenized POS-tagged and parsed text

Uses:
    Text preprocessing
    Keyword in Context
    Topic modeling
    Information Extraction
    Keyterm extraction
    Text and readability statistics
    Emotional valence analysis
    Quotation attribution
"""

import textacy

raw_text = """When updating to a newer version of spaCy, it's generally recommended to start with a clean virtual environment. 
If you're upgrading to a new major version, make sure you have the latest compatible models installed, and that there 
are no old shortcut links or incompatible model packages left over in your environment, as this can often lead to 
unexpected results and errors. If you've trained your own models, keep in mind that your train and 
runtime inputs must match. This means you'll have to retrain your models with the new version.
Donate $40. venkatram@gmail.com, This is taken from https://spacy.io/usage/"""

#Removing Punctuation and Uppercases
modified_text = textacy.preprocess.remove_punct(raw_text)
print(modified_text)

print("*****************")
modified_text = textacy.preprocess.replace_urls(raw_text, replace_with="SpaCy")
print(modified_text)

print("*****************")
modified_text = textacy.preprocess.replace_currency_symbols(raw_text, replace_with="USD")
print(modified_text)

