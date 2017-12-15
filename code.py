'''
CME594 Introduction to Data Science
Homework 9 Code - Neural Networks and Deep Learning
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing #to normalize the values
from sklearn.model_selection import train_test_split #new for version 0.18 but seems to be out soon
import matplotlib.pyplot as plt
import keras.callbacks
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dense, Dropout
from keras.layers.recurrent import LSTM

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'data'
input_data = pd.read_csv(file_name + '.csv', header=0, index_col=0)
input_data= input_data.apply(pd.to_numeric, errors='coerce')
input_data=input_data.dropna()
analysis_type = 'R'

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")

#Defining X1, X2, and all the data X
X1 = input_data.DJI.values.astype(float)
X2 = input_data.MSH.values.astype(float)
X3 = input_data.XMI.values.astype(float)
X4 = input_data.INX.values.astype(float)
X5 = input_data.XLF.values.astype(float)
X6 = input_data.IXIC.values.astype(float)
X7 = input_data.ConSen.values.astype(float)
X8 = input_data.MZM.values.astype(float)
X9 = input_data.spread.values.astype(float)

X_raw = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8,X9))

#Normalizing or not the data - but normalizes all data, not column by column
X =preprocessing.normalize(X_raw) #does not seem to improve the accuracy
#X = X_raw

#Defining Y variables depending on whether we have a regression or classification problem
Y = input_data.NUE.values.astype(float)


##Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

sc=[]
ind=[]
 
fig=plt.figure(figsize=(7,9))
count=0
for i in range(1,21,1):
     count=count+1
     for k in range(1,100,1):
         if analysis_type == 'R':
             #Fit the neural network for Regression purposes (i.e., you expect a continuous variable out)
             #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
             acti = ['logistic', 'tanh', 'relu']
             algo = ['lbfgs', 'sgd', 'adam']
             learn = ['constant', 'invscaling', 'adaptive']
             neural = MLPRegressor(activation=acti[2], solver=algo[0], batch_size = 1, learning_rate = learn[1], hidden_layer_sizes=(i,)) 
             neural.fit(X_train, Y_train)
             neural_score = neural.score(X_test, Y_test)
             sc.append(neural_score)
             ind.append(k)
             #        print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
             #        print("Coefs: ")
             #        print(neural.coefs_[0].round(2))
             #        print(neural.coefs_[1].round(2))
             #        print("Intercepts: {0}".format(neural.intercepts_))
             #        print("Iteration: {0}".format(neural.n_iter_))
             #        print("Layers: {0}".format(neural.n_layers_))
             #        print("Outputs: {0}".format(neural.n_outputs_))
             #        print("Activation: {0}".format(neural.out_activation_))
             #        
             #        #Assess the fitted Neural Network
             #        print("Y test and predicted")
             #        print(Y_test.round(1))
             #        print(neural.predict(X_test).round(1))
             #        print("")
             #        print("Accuracy as Pearson's R2: {0}".format(neural_score.round(4)))
             
        
                                                                     
         plt.subplot(5,4,count)
         plt.plot(ind,sc, 'o')
         x1,x2,y1,y2 = plt.axis()
         plt.axis((x1,x2,0,1))
         plt.gca().set_xticks([])
 
plt.tight_layout()
plt.xlabel("iteration number")
plt.ylabel("score")
fig.savefig('back.png',format='png', dpi=300)
#==============================================================================
## Call back to capture losses 
#class LossHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.losses = []
#
#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        
## You should get data frames with prices somewhere, e.g. on Quandl - implementation is up to you
## merge data frames
## m1 = Y.merge(X1, left_index=True, right_index=True, how='inner')
#
#
## data prep
## use 100 days of historical data to predict 10 days in the future
#data = input_data.values
#examples = 100
#y_examples = 7
#nb_samples = len(data) - examples - y_examples
#
## input - 2 features
#input_list = [np.expand_dims(np.atleast_2d(data[i:examples+i,:]), axis=0) for i in xrange(nb_samples)]
#input_mat = np.concatenate(input_list, axis=0)
#
#
## target - the first column in merged dataframe
#target_list = [np.atleast_2d(data[i+examples:examples+i+y_examples,0]) for i in xrange(nb_samples)]
#target_mat = np.concatenate(target_list, axis=0)
#
#
## set up model
#trials = input_mat.shape[0]
#features = input_mat.shape[2]
#hidden = 128
#model = Sequential()
#model.add(LSTM(hidden, input_shape=(examples, features)))
#model.add(Dropout(.2))
#model.add(Dense(y_examples))
#model.add(Activation('relu'))
#model.compile(loss='mse', optimizer='rmsprop')
#
# # Train
#history = LossHistory()
#model.fit(input_mat, target_mat, nb_epoch=100, batch_size=400, callbacks=[history])
#
#




