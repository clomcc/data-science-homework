'''
CME594 Introduction to Data Science
Homework 11 Code - Text Mining with TextBlob
Marina Corby
'''

#Libraries needed to run the tool
import numpy as np
from nltk import *
#from nltk.book import *
from textblob import TextBlob
import matplotlib.pyplot as plt 


open_abstract = open('original_abstract.txt', 'rU')
read_abstract = open_abstract.read()
abstract = TextBlob(read_abstract)

#Problem 1
'''
print read_abstract
'''

#Problem 4
#iii. Sentiment of full text

print("Sentiment of abstract: {0}".format(abstract.sentiment))


#iv. Sentiment of sentences

print("Sentiment of each sentence in abstract:")
print("")
for sent in abstract.sentences:
    #print("Sentiment of sentence '{0}':".format(sent)) #prints each sentence
    print(sent.sentiment)
    print("")


#v. polarity vs subjectivity

polarity_list = []
subjectivity_list = []

for sent in abstract.sentences:
    polarity_list.append(sent.polarity)
    subjectivity_list.append(sent.subjectivity)


plt.plot(polarity_list, subjectivity_list, 'r*')
plt.show()


