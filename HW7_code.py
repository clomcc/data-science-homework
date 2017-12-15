'''
CME594 Introduction to Data Science
Homework 7 Code - Support Vector Machine
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'class_grades'
input_data = pd.read_csv(file_name + '.csv', header=0)


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")

#Defining X1, X2, and Y
X = np.column_stack((input_data["X1"].values, input_data["X2"].values))
Y = input_data["Y"].values

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#Setting up SVM and fitting the model with training data
ker = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] #we won't use precomputed
scores=[]


vector = svm.SVC(kernel=ker[i], degree=3) #degree is only relevant for the 'poly' kernel
    vector.fit(X_train, Y_train)
    Y_predict = vector.predict(X_test)
    print Y_predict
    print Y_test
    print metrics.accuracy_score(Y_predict, Y_test)
    scores.append(metrics.accuracy_score(Y_predict,Y_test))

n=[1,2,3,4]
Bar=plt.bar(n, scores, width=.4, color='#669AC0', tick_label=ker[0:4], align='center')
plt.ylabel('Accuracy Score', fontsize=14, fontname='fantasy')
plt.xlabel('Kernel Function', fontsize=14, fontname='fantasy')
plt.setp(Bar.get_xticklabels(), fontsize=12, fontname='fantasy')

plt.show()

'''
print("Indices of support vectors:")
print(vector.support_)
print("")
print("Support vectors:")
print(vector.support_vectors_)
print("")
print("Intercept:")
print(vector.intercept_)
print("")

if vector.kernel == 'linear':
    c = vector.coef_
    print("Coefficients:")
    print(c)
    print("This means the linear equation is: y = -" + str(c[0][0].round(2)) + "/" + str(c[0][1].round(2)) + "*x + " + str(vector.intercept_[0].round(2)) + "/" + str(c[0][1].round(2)))
    print("")


#Run the model on the test (remaining) data and show accuracy
Y_predict = vector.predict(X_test)
print Y_predict
print Y_test
print metrics.accuracy_score(Y_predict, Y_test)


#Adapted from http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5 #Defines min and max on the x-axis
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5 #Defines min and max on the y-axis
h = (x_max - x_min)/300 # step size in the mesh to plot entire areas
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Defines meshgrid
Z = vector.predict(np.c_[xx.ravel(), yy.ravel()]) #Uses the calibrated model knn and running on "fake" data in meshgrid

# Put the result into a color plot
Z = Z.reshape(xx.shape) #Reshape for matplotlib
plt.figure(1) #create one figure
plt.set_cmap(plt.cm.Paired) #Picks color for 
plt.pcolormesh(xx, yy, Z) #Plot for the data

# Plot also the training points
colormap = np.array(['white', 'black']) #BGive two colors based on values of 0 and 1 from HW6_Data
plt.scatter(X[:,0], X[:,1],c=colormap[Y]) #Plot the data as a scatter plot, note that the color changes with Y.

plt.xlabel("X1") #Adding axis labels
plt.ylabel("X2")
plt.xlim(xx.min(), xx.max()) #Setting limits of axes
plt.ylim(yy.min(), yy.max())
plt.xticks(()) #Removing tick marks
plt.yticks(())
plt.savefig(file_name + '_plot.png') #Saving the plot

plt.show() #Showing the plot




