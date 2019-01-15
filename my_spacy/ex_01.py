import spacy

nlp = spacy.load('en')

doc = nlp(u"Tesla is looking at buying U.S. startup for $ million")

for token in doc:
    print(token.text, token.pos_,token.dep_)

print(spacy.explain("nsubj"))

print(nlp.pipeline)
print(nlp.pipe_names)
print("******************")

doc2 = nlp(u"Tesla isn't looking into startup for anymore")

for token in doc2:
    print(token.text, token.pos_,token.dep_)

print("***********************")
doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')
quote = doc3[16:30]
print(quote)

print(type(quote))

print("***********************")
doc4 = nlp(u"This is Ram. He is a software developer. He works in a company for living.")
for sentence in doc4.sents:
    print(sentence)