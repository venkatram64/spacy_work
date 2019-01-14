from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter
import spacy
import json

"""
Intent Classification with Rasa NLU Spacy

    A Library for intent recognition and entity extraction based on Spacy and Sklearn
    NLP=NLU+NLG+More
    
        NLP = understand, process, interprete everyday human language
        NLU = unstructured inputs and convert them into a structured from that
        a machine can understand
    
    Uses:
        Chatbot tast
        NL Understanding
        Intent classification
        
    installation
        pip install rasa_nlu
        python -m rasa_nlu.server & sklearn_crfsuite
        
    ******* OR **************    
    using spacy as backend
        pip3.7 install rasa_nlu[spacy]
        python -m spacy download en_core_web_md
        python -m spacy link en_core_web_md en
"""

train_data = load_data("./data/rasa_dataset.json")

#Config Backend using Sklearn and spacy

trainer = Trainer(config.load("config_spacy.json"))

#Training Data
t_data = trainer.train(train_data)

print(t_data)

model_dir = trainer.persist('./models/nlu', fixed_model_name='predict')
#Entity Extraction
nlp = spacy.load('en')
docx = nlp(u"I am looking for an Italian Restaurant where I can eat.")

for word in docx.ents:
    print("value: ", word.text, " entity: ", word.label_, " start: ",word.start_char," end: ", word.end_char)

"""
Making Predictions with Model
Interpreter.load().parse()
"""

interpreter = Interpreter.load('./models/nlu/default/predict')

#Prediction of Intent
result = interpreter.parse(u"I am looking for an Italian Restaurant where I can eat.")

print(json.dumps(result, indent=2))

result = interpreter.parse(u"I want an African Spot to eat.")

print(json.dumps(result, indent=2))

