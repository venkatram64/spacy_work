import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Semantic Similiarty
object1.similarity(object2)
Uses
For recommendation systems
Data preprocessing eg removing duplicates
python -m spacy download encore_web_lg
"""

nlp = spacy.load('en')

doc1 = nlp("wolf")
doc2 = nlp("dog")
print(doc1.similarity(doc2))

print("*******************")
doc3 = nlp("cat")
print(doc3.similarity(doc2))

print("*******************")
# Synonyms
doc1 = nlp("smart")
doc2 = nlp("clever")
print(doc1.similarity(doc2))

print("*******************")
ex1 = nlp("wolf dog cat bird fish")
for token1 in ex1:
    for token2 in ex1:
        print((token1.text, token2.text), "Similarity ", token1.similarity(token2))

print("*******************")
my_list = [(token1.text, token2.text, token1.similarity(token2)) for token2 in ex1 for token1 in ex1]

print(my_list)

df = pd.DataFrame(my_list)
print(df.head())

print(df.corr())

df.columns = ['Token1', 'Token2', 'Similarity']
print(df.head())

df_viz = df.replace({'wolf':0, 'dog':1, 'cat':2, 'bird':3, 'fish':4})

#Plotting with Correlation
plt.figure(figsize=(20,10))
sns.heatmap(df_viz.corr(), annot=True)
plt.show()

#Plotting with no correlation
plt.figure(figsize=(20,10))
sns.heatmap(df_viz, annot=True)
plt.show()