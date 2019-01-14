import spacy


"""
Noun Chunks
    noun + word describing the noun
    noun phrases
    adnomial
    root.text
"""

nlp = spacy.load('en')
docx = nlp(u"The man reading the news is very tall.")

for token in docx.noun_chunks:
    print(token.text)


#Root Text
#The Main Nound to the rest
print("*******Root Text***********")
for token in docx.noun_chunks:
    print(token.root.text)

#Text of the root token head
print("*******Root Token Head***********")
for token in docx.noun_chunks:
    print(token.root.text, " Connector_Text: ", token.root.head.text)