import spacy

nlp = spacy.load('en')

doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today.")

def show_lemmas(text):

    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}}  {token.lemma_}')


show_lemmas(doc1)