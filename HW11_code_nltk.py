'''
CME594 Introduction to Data Science
Homework 11 Code - Text Mining with NLTK
Marina Corby
'''

#Libraries needed to run the tool
import numpy as np
import nltk as nltk
from nltk import *
import matplotlib.pyplot as plt
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer

#f=open('original_abstract.txt','rU')
#raw=f.read()
doc="Final Project Abstract: Predicting the Effects of the Great Recession on Stock Prices This project is designed to develop a model for predicting stock prices. This type of analysis has been done many times, but I am planning to attempt to predict the effect of a black swan event on stock prices using data before the event occurs. In particular, I want to focus on the stock market crash in October of 2008 following the collapse of Lehman Brothers. This collapse had an immediate effect on stock prices across the board, which makes me curious if it is possible to predict the random collapse of stock prices outside the financial industry. My data will be historical stock prices, which will come from Google Finance. In particular, my plan is to focus on stocks in the steel industry. These stocks collapsed with the rest of the economy but lead the recovery, which indicates that they have an interesting relationship to the economy as a while. I may try to find some other data, such as number of news stories about manufacturing projects, which is not directly visible in historical stock prices but greatly impacts them. Neural networks and Support Vector Machine are both standard techniques for predicting stock movements and prices. These two methods are useful in this setting for distinct reasons, which will make it interesting to compare their results. Neural networks are very good for analyzing and making predictions for data with complicated and opaque relationships between the inputs. Buying and selling stock is (sometimes) profitable because the economy is sufficiently complicated (and has enough random variation as a result of the human involvement) that no one is able to fully understand the relationships. Thus, the fact that the output of a neural network is hard to interpret seems fitting in this setting. However, stock prices tend to be very noisy data, which means that a neural network might have an overfitting problem. This is where support vector machine comes in as a desirable technique. It is a great model for noisy data, so it may do a better job of picking up trends rather than random variation."
raw=doc.lower()
doc2="cat in hat"
raw2=doc2.lower()

#text = nltk.Text(tokens)
#sentences = nltk.sent_tokenize(raw)


#Problem 1 - print abstract
'''
print raw
'''

#Problem 2
'''
print("i.   tokens : {0}".format(len(raw)))
print("ii.  unique tokens : {0}".format(len(set(raw))))
print("iii. number of words : {0}".format(len(tokens)))
print("iv.  unique words : {0}".format(len(set(tokens))))
print("v.   sentences : {0}".format(len(sentences)))
print("vi.  unique sentences : {0}".format(len(set(sentences))))
'''

#Problem 3
'''
print(text.concordance("event"))
print("")
print("Similar to 'event' in abstract:")
print(text.similar("event"))
print("")
fdist = FreqDist(text)
print("Number of times 'event' is present: {0}".format(text.count('event')))
#print("Number of times 'predict' is present: {0}".format(fdist["predict"]))
print("Frequency (i.e., between 0 and 1) of the word 'event': {0}".format(fdist.freq("event")))
'''

#Problem 4 (parts iii-v are in textblob code)
# i. frequency distribution

#Plot the frequency distribution
'''
fdist1 = FreqDist(text)
plt.figure(figsize=(3,3))
fdist1.plot(50, cumulative=True)

#ii. dispersion plot 

text.dispersion_plot(["stock", "data", "event"])
'''
tokenizer = RegexpTokenizer(r'\w+')
tokens = nltk.tokenize(raw)
tokens2=nltk.tokenize(raw2)
en_stop = get_stop_words('en')
stopped_tokens = [i for i in tokens if not i in en_stop]
stopped_tokens2=[j for j in tokens2 if not j in en_stop]
#p_stemmer = PorterStemmer()
#texts = [p_stemmer.stem(i) for i in stopped_tokens, p_stemmer.stem(j) for j in stopped_tokens2]
texts=[stopped_tokens,stopped_tokens2]
dictionary = corpora.Dictionary([texts])
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=3))