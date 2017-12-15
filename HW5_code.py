'''
CME594 Introduction to Data Science
Homework 5 Code - k-Nearest Neighbor Algorithm
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from time import strptime

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'forestfires'
input_data = pd.read_csv(file_name + '.csv', header=0)


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")

#Defining X1, X2, and Y
X = np.column_stack((input_data.temp.values, input_data.RH.values))
input_data["monthindex"] = input_data['month'].apply(lambda x: strptime(x, '%b').tm_mon)
Y = input_data.monthindex.values

#Select training data from our entire data set
#X_train = X[:15,:]
#Y_train = Y[:15]

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Make the loop for k from 1 - 60
acclist = np.zeros(61)
for k in range(1, 61):  
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(X_train, Y_train)

    #Run the model on the test (remaining) data and show accuracy
    Y_predict = knn.predict(X_test)
    acc = metrics.accuracy_score(Y_predict, Y_test)
    acclist[k] = acc

print(np.max(acclist))
print(np.argmax(acclist))

plt.plot(acclist) 
plt.savefig('problem 3')
plt.show()


#Setting up k-Nearest Neighbor and fitting the model with training data
k = np.argmax(acclist)
print(k)
knn = neighbors.KNeighborsClassifier(k)
knn.fit(X_train, Y_train)

#Run the model on the test (remaining) data and show accuracy
#X_test = X[15:,:]
#Y_test = Y[15:]
Y_predict = knn.predict(X_test)
print Y_predict
print Y_test
print metrics.accuracy_score(Y_predict, Y_test)


#Adapted from http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5 #Defines min and max on the x-axis
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5 #Defines min and max on the y-axis
h = (x_max - x_min)/200 # step size in the mesh to plot entire areas
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Defines meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]) #Uses the calibrated model knn and running on "fake" data in meshgrid

# Put the result into a color plot
Z = Z.reshape(xx.shape) #Reshape for matplotlib
plt.figure(1) #create one figure
plt.set_cmap(plt.cm.Paired) #Picks color for 
plt.pcolormesh(xx, yy, Z) #Plot for the data

# Plot also the training points
colormap = np.array(['white', 'black']) #BGive two colors based on values of 0 and 1 from HW6_Data
plt.scatter(X[:,0], X[:,1],c=colormap[Y]) #Plot the data as a scatter plot, note that the color changes with Y.
plt.title('Nearest Neighbors Plot')
plt.xlabel("Variable 1") #Adding axis labels
plt.ylabel("Variable 2")
plt.xlim(xx.min(), xx.max()) #Setting limits of axes
plt.ylim(yy.min(), yy.max())
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
fig.Figure(figsize='5,5')
plt.savefig(file_name + '_plot.png') #Saving the plot

plt.show() #Showing the plot
