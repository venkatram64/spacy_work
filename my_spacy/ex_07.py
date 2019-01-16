import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en')

with open('owlcreek.txt') as f:
    doc = nlp(f.read())

print(doc[:36])

tokens = [token for token in doc]

print(len(tokens))

sentences = [sentennce for sentennce in doc.sents]

print(len(sentences))

print(sentences[1])
print("****************")

for token in sentences[1]:
    print(token.text, token.pos_, token.dep_, token.lemma_)

print("**********************")

for token in sentences[1]:
    print(f'{token.text:{12}} {token.pos_:{6}} {token.dep_:{8}}  {token.lemma_}')

matcher = Matcher(nlp.vocab)


#Swimming vigorously
pattern = [{'LOWER': 'swimming'}, {'IS_SPACE': True, 'OP': '*'}, {'LOWER': 'vigorously'}]

#adding pattern to matcher

matcher.add('Swimming', None, pattern)

found_matches = matcher(doc)

print(found_matches)
print("***************")
def surrounding(doc, start, end):
    print(doc[start - 5: end + 5])
print("***************")
surrounding(doc, 1274, 1277)

surrounding(doc, 3607, 3610)
print("***************")
for sentence in sentences:
    if found_matches[0][1] < sentence.end:
        print(sentence)
        break

print("***************")
for sentence in sentences:
    if found_matches[1][1] < sentence.end:
        print(sentence)
        break



