import spacy

"""
Sentence Segmentation or Boundary Detection

Deciding where sentences begin and end

    1. If it's a period, it ends a sentence.
    2. If the preceding token is in the hand-compiled list of abbreviations, then it does not end sentence.
    3. If the next token is capitalized, then it ends a sentence.
    
Default = Uses the dependency Manual
    You set boundaries before parsing the doc.
"""
nlp = spacy.load('en')
#Custom / Manual Function

def my_custom_boundary(docx):
    for token in docx[:-1]:
        if token.text == '---':
            docx[token.i + 1].is_sent_start = True
    return docx

#Adding the rule before parsing

nlp.add_pipe(my_custom_boundary, before='parser')

my_sentence = nlp(u"This is my first sentence---the last comment was so cool---what if---? this the last sentence")

for sentence in my_sentence.sents:
    print(sentence)

#Sentence
#This

print("*************")

nlp = spacy.load('en')

my_sentence = nlp(u"This is my first sentence---the last comment was so cool---what if---? this the last sentence")

for sentence in my_sentence.sents:
    print(sentence)