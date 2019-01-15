import spacy
from spacy import displacy

nlp = spacy.load('en')

my_string = '"We\'re moving to L.A.!"'

doc = nlp(my_string)

for token in doc:
    print(token.text)

print("****************")
doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com!")

for token in doc2:
    print(token.text)

print("****************")
doc3 = nlp(u"A 5km NYC cab ride cost $10.5")

for token in doc3:
    print(token.text)

print("length: ", len(doc3))

print("Vocabilary: ", len(doc3.vocab))

doc4 = nlp(u"Apple to build a Hong Kong factory for $6 million")

for token in doc4:
    print(token.text, end=' | ')

print('\n')

for entity in doc4.ents:
    print(entity, entity.label_, '\n')
    print(str(spacy.explain(entity.label_)), '\n')

doc5 = nlp(u"Autonomous cars shift insurance liability toward manufacturers")
for chunk in doc5.noun_chunks:
    print(chunk)

doc6 = nlp(u"Apple is going to build a U.K. factory for $6 million")
#displacy.serve(doc6, style='dep', options={'distance':110})

doc7 = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for profit of 2 millions.")
displacy.serve(doc7, style='ent', options={'distance':110})