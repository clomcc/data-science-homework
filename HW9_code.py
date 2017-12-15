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

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'forestfires'
input_data = pd.read_csv(file_name + '.csv', header=0)

analysis_type = 'C'

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")

#Defining X1, X2, and all the data X
X1 = input_data.FFMC.values.astype(float)
X2 = input_data.DMC.values.astype(float)
X3 = input_data.DC.values.astype(float)
X4 = input_data.ISI.values.astype(float)
X5 = input_data.temp.values.astype(float)
X6 = input_data.RH.values.astype(float)
X7 = input_data.wind.values.astype(float)
X8 = input_data.rain.values.astype(float)
X_raw = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8))

#Normalizing or not the data - but normalizes all data, not column by column
#X =preprocessing.normalize(X_raw) #does not seem to improve the accuracy
X = X_raw

#Defining Y variables depending on whether we have a regression or classification problem
if analysis_type == 'R':
    Y = input_data.area.values.astype(float)
else:
    Y = input_data.month.values

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
sc=[]
ind=[]

for k in range(1,101,1):
    if analysis_type == 'R':
        #Fit the neural network for Regression purposes (i.e., you expect a continuous variable out)
        #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
        acti = ['logistic', 'tanh', 'relu']
        algo = ['lbfgs', 'sgd', 'adam']
        learn = ['constant', 'invscaling', 'adaptive']
        neural = MLPRegressor(activation=acti[2], solver=algo[0], batch_size = 1, learning_rate = learn[2], hidden_layer_sizes=(k,)) 
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
        
    else:
      #Fit the neural network for Classification purposes (i.e., you don't expect a continuous variable out).
      #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
      acti = ['logistic', 'tanh', 'relu']
      algo = ['lbfgs', 'sgd', 'adam']
      learn = ['constant', 'invscaling', 'adaptive']
      neural = MLPClassifier(activation=acti[2], solver=algo[1], batch_size = 1, learning_rate = learn[2], hidden_layer_sizes=(7,7)) 
      neural.fit(X_train, Y_train)
      neural_score = neural.score(X_test, Y_test)
      
      sc.append(neural_score)
      ind.append(k)
#    print("Classes: {0}".format(neural.classes_))
#    print("")
#    print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
#    print("")
#    print("Coefs: ")
#    print(neural.coefs_[0].round(2))
#    print("")
#    print(neural.coefs_[1].round(2))
#    print("")
#    print("Intercepts: {0}".format(neural.intercepts_))
#    print("")
#    print("Iteration: {0}".format(neural.n_iter_))
#    print("")
#    print("Layers: {0}".format(neural.n_layers_))
#    print("")
#    print("Outputs: {0}".format(neural.n_outputs_))
#    print("")
#    print("Activation: {0}".format(neural.out_activation_))

    #Assess the fitted Neural Network
#    print("Y test and predicted")
#    print(Y_test)
#    print(neural.predict(X_test))
#    print("")
#    print("Mean Accuracy: {0}".format(neural_score.round(4)))

plt.plot(ind,sc)
plt.xlabel("number of nodes")
plt.ylabel("score")
    