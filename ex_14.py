
from gensim import models
import logging

#https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
"""
GenSim => Generate Similar
What is Topic Modelling?
It is the process of extracting topics from large amount of documents.
Used frequently in text mining tool.
It is also used in soft assignment of topic(Modern Topic modelling)
Way to anaalyse large volume of unlabelled text
Cluster of similar words
Probabilistic topic modelling
It can work on continuous in coming data.
Categories
Baseic Methods                            Topic Evolution Model

    1. Latent Semantic Analysis(LSA)      1. Topic Over Time(TOT)
    2. Probabilistic Latent Semantic      2. Dynamic Topic Models(DTM)
       Aanalysis(PLSA)
    3. Latent Dirishlet Allocation(LDA)   3. Multiscale Topic Tomography(MTT)
    4. Correlated Topic Model(CTM)        4. Dynamic Topic Correlation Detection(DTCD)
                                          5. Detecting Topic Evolution(DTE)

    Core Concepts:
    Corpus
    Vector
    Model

"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set("for a lot of the and to in".split(" "))

texts = [[word for word in document.lower().split() if word not in stoplist] for document in raw_corpus]


model = models.Word2Vec(texts, min_count=1)
print(model.wv.index2word)

#model.save('test_model')

#model = gensim.models.word2vec.load('test_model')
print(model.most_similar("human"))