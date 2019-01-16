import spacy

nlp = spacy.load('en')

print(len(nlp.Defaults.stop_words))

print(nlp.Defaults.stop_words)

print(nlp.vocab['is'].is_stop)
print(nlp.vocab['mystory'].is_stop)

print("Adding custom stop words")
#btw -> by the way
nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True
print(nlp.vocab['btw'].is_stop)

#removing the defaults stop words from spacy
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False
print(nlp.vocab['beyond'].is_stop)
