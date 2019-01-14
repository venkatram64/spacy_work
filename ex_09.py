from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm

#Training & Updating Our Named Entity Recognizer
"""
Training our NER (named entity recognizer)
Updating our NER
"""

nlp1 = spacy.load('en')
docx1 = nlp1(u"Who was Kofi Annan?")

for token in docx1.ents:
    print(token.text, token.start_char, token.end_char, token.label_)



TRAIN_DATA = [
    ("Who is Kofi Annan?",{
        "entities":[(8, 18, "PERSOn")]
    }),
    ("Who is Steve Jobs?",{
        "entities":[(8, 17, "PERSOn")]
    }),
    ("I like London and Berlin",{
        "entities":[(7, 13, "LOC"),(18, 24, "LOC")]
    })
]

##Annotate
plac.annotations(
    model=("Model name, Defaults to blank. 'en' model.", "option", "m", str),
    output_dir=("C:\\Users\\Venkatram\\Documents\\JLaba\\JFlow", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int)
   )

#Define our varibales

model = None
output_dir = Path("C:\\Users\\Venkatram\\Documents\\JLaba\\JFlow")
n_iter = 100

#Load the model

if model is not None:
    nlp = spacy.load(model)
    print("Loadded mode {}".format(model))
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

#Set Up the Pipeline
"""
create the built in pipeline components and add them to the pipline
nlp.create_pipe works for build ins that are registered with spacy
"""
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pip('ner')
    nlp.add_pip(ner, last=True)
    #otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe('ner')

"""
Train the Recogniser
Add labels, Annotate them
Pipes
Begin training
"""

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_labels(ent[2])


#get names of other pipes ot disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_trining()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],
                [annotations],
                drop=0.5,
                sgd=optimizer,
                losses=losses
            )
        print(losses)

#Test the trained mode

for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print("Entities ",[(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens ", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

#Save the model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to ", output_dir)


#Test the saved model
#NB Ouput Directory

print("Loading from ", output_dir)
nlp2 = spacy.load(output_dir)

for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print("Entities ", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens ", [(t.text, t.ent_type_, t.ent_iob) for t in doc])