import spacy


nlp = spacy.load('en')

doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

print(doc.text)  #print(doc)

for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

print("*************************")

#working with POS Tags

doc = nlp(u'I read books on NLP.')
for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

print("*************************")
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

POS_counts = doc.count_by(spacy.attrs.POS)

print(POS_counts)

print(doc.vocab[83].text)
print("********************")

#Create a frequency list of POS tags

for k, v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')

print("********************")
#count the different fine grained tags:
TAG_counts = doc.count_by(spacy.attrs.TAG)

for k, v in sorted(TAG_counts.items()):
    print(f'{k:{20}}. {doc.vocab[k].text:{4}}: {v}')

print("********************")

#Count the different dependencies:
DEP_counts = doc.count_by(spacy.attrs.DEP)
for k, v in sorted(DEP_counts.items()):
    print(f'{k:{20}}. {doc.vocab[k].text:{4}}: {v}')



