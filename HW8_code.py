'''
CME594 Introduction to Data Science
Homework 8 Code - Decision Tree Learning
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'HW8_data'
input_data = pd.read_csv(file_name + '.csv', header=0, index_col=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")

#Defining X1, X2, and all the data X
X0 = input_data.AvgHW.values.astype(int)
X1 = input_data.AvgQuiz.values.astype(int)
X2 = input_data.AvgLab.values.astype(int)
X3 = input_data.MT1.values.astype(int)
X4 = input_data.MT2.values.astype(int)
X5 = input_data.Final.values.astype(int)
X = np.column_stack((X0, X1, X2, X3, X4, X5))
Y = input_data.Grade.values

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Fit the Decision tree
crit_choice = ['gini', 'entropy']
crit = crit_choice[1]
decitree = tree.DecisionTreeClassifier(criterion=crit).fit(X_train, Y_train) #Decision Tree
randfor = RandomForestClassifier(n_estimators = 2, criterion=crit).fit(X_train, Y_train) #Random Forest

#Export tree properties in graph format
#to see the graph need to use 'dot -Tpng HW8_data.dot -o HW8_data.png' in command prompt, if needed, install Graphviz from http://www.graphviz.org/
#alternatively, copy and paste the text from the .dot file to http://www.webgraphviz.com/
tree.export_graphviz(decitree, out_file= file_name + '.dot')

decitree_predict = decitree.predict(X_test)
decitree_score = metrics.accuracy_score(decitree_predict, Y_test)

randfor_predict = randfor.predict(X_test)
randfor_score = metrics.accuracy_score(randfor_predict, Y_test)

print Y_test
print("Decision Tree")
print decitree_predict
print decitree_score
print("Random Forest")
print randfor_predict
print randfor_score
