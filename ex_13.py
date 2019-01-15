from collections import defaultdict
from gensim import corpora, models, similarities
import logging
import tempfile
import os.path


#https://www.youtube.com/watch?v=GmqRt1vHcz8&index=2&list=PL7zYW7FHOfhgfFnfWXSFecQLU5bhUOR5j
#gensim Quick Start
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim%20Quick%20Start.ipynb
#Topics_and_Transformations.ipynb
#https://github.com/RaRe-Technologies/gensim/blob/f3cf463c0f0e28c97c9a3b319a58a7e099092041/docs/notebooks/Topics_and_Transformations.ipynb
#Similarity_Queries
#https://github.com/RaRe-Technologies/gensim/blob/a864e0247dda421d8e4de280fa91f86f474a5691/docs/notebooks/Similarity_Queries.ipynb
#https://radimrehurek.com/gensim/tut3.html

"""
Corpus: A Copus is a collection of digital documents
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

#Create a set of frequest words

stoplist = set("for a lot of the and to in".split(" "))

#Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist] for document in raw_corpus]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

## Only keep words that appear more than once
processed_corupus = [[token for token in text if frequency[token] > 1] for text in texts]
print(processed_corupus)

dictionary = corpora.Dictionary(processed_corupus)
dictionary.save("/tmp/deerwester.dict")
print(dictionary)

"""
Vector
To infer the latent structure in our corpus we need a way to represent documents that we can manipulate mathematically. 
One approach is to represent each document as a vector. There are various approaches for creating 
a vector representation of a document but a simple example is the bag-of-words model. Under the bag-of-words model 
each document is represented by a vector containing the frequency counts of each word in the dictionary. For example, 
given a dictionary containing the words ['coffee', 'milk', 'sugar', 'spoon'] a document consisting of the string 
"coffee milk coffee" could be represented by the vector [2, 1, 0, 0] where the entries of the vector are (in order)
 the occurrences of "coffee", "milk", "sugar" and "spoon" in the document. The length of the vector is the number of 
 entries in the dictionary. One of the main properties of the bag-of-words model is that it completely ignores the 
 order of the tokens in the document that is encoded, which is where the name bag-of-words comes from.
Our processed corpus has 12 unique words in it, which means that each document will be represented by a 12-dimensional 
vector under the bag-of-words model. We can use the dictionary to turn tokenized documents into these 12-dimensional 
vectors. We can see what these IDs correspond to:
"""

print(dictionary.token2id)

new_doc = "Human computer interaction"
new_doc = dictionary.doc2bow((new_doc.lower().split()))
print(new_doc)

corpus = [dictionary.doc2bow(text) for text in processed_corupus]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
print(corpus)

"""
Model
Now that we have vectorized our corpus we can begin to transform it using models. We use model as an abstract term 
referring to a transformation from one document representation to another. In gensim documents are represented as 
vectors so a model can be thought of as a transformation between two vector spaces. The details of this transformation 
are learned from the training corpus.
One simple example of a model is tf-idf. The tf-idf model transforms vectors from the bag-of-words representation to a 
vector space where the frequency counts are weighted according to the relative rarity of each word in the corpus.
Here's a simple example. Let's initialize the tf-idf model, training it on our corpus and transforming the string "system minors":
"""

#train the model
tfidf = models.TfidfModel(corpus)
"""
in case of TfIdf, the “training” consists simply of going through the supplied corpus once and computing document 
frequencies of all its features. Training other models, such as Latent Semantic Analysis(LSA) or 
Latent Dirichlet Allocation(LDA), is much more involved and, consequently, takes much more time.
"""
print(tfidf[dictionary.doc2bow("system minors".lower().split())])

"""
The tfidf model again returns a list of tuples, where the first entry is the token ID and the second entry is the 
tf-idf weighting. Note that the ID corresponding to "system" (which occurred 4 times in the original corpus) has been 
weighted lower than the ID corresponding to "minors" (which only occurred twice).
"""

TEMP_FOLDER = tempfile.gettempdir()
print("Folder '{}' will be used to save temporary dictionary and corpus.".format(TEMP_FOLDER))

if os.path.isfile(os.path.join(TEMP_FOLDER, 'deerwester.dict')):
    dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'deerwester.dict'))
    corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, "deerwester.mm"))
    print("Used files generated from first tutorial. ")
else:
    print("Please run first tutorial to generate data set")



print(dictionary[0])
print(dictionary[1])
print(dictionary[2])

#Creating a transformation

tfidf = models.TfidfModel(corpus)
doc_bow = [(0,1), (1,1)]
print(tfidf[doc_bow])

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

# initialize an LSI transformation
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsi = lsi[corpus_tfidf]

"""
Here we transformed our Tf-Idf corpus via Latent Semantic Indexing into a latent 2-D space (2-D because we set num_topics=2). 
Now you’re probably wondering: what do these two latent dimensions stand for? Let’s inspect with models.LsiModel.print_topics():
"""

print(lsi.print_topics(2))

# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly

for doc in corpus_lsi:
    print(doc)

# same for tfidf, lda, ...
#lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER, 'model.lsi'))

lsi.save(os.path.join(TEMP_FOLDER, 'model.lsi'))

"""
Initializing query structures

To prepare for similarity queries, 
we need to enter all documents which we want to compare against subsequent queries. 
In our case, they are the same nine documents used for training LSI, converted to 2-D LSA space. 
But that’s only incidental, we might also be indexing a different corpus altogether.
"""
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])
#index = similarities.MatrixSimilarity.load(os.path.join(TEMP_FOLDER, 'index'))
index.save(os.path.join(TEMP_FOLDER, 'deerwester.index'))

# perform a similarity query against the corpus
sims = index[vec_lsi]
print(list(enumerate(sims)))

"""
With some standard Python magic we sort these similarities into descending order, 
and obtain the final answer to the query “Human computer interaction”:
"""

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)