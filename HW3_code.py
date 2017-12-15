'''
CME594 Introduction to Data Science
Homework 3 Code - Principal Component Analysis (PCA)
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
np.set_printoptions(suppress=True, precision=5, linewidth=150) #to control what is printed: 'suppress=True' prevents exponential prints of numbers, 'precision=5' allows a max of 5 decimals, 'linewidth'=150 allows 150 characters to be shown in one line (thus not cutting matrices)
import pandas as pd
from sklearn.decomposition import PCA
#from sklearn.preprocessing import LabelEncoder #To switch categorical letters to numbers
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'usa_00006'
input_data = pd.read_csv(file_name + '.csv', header=0)



#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")


#Defining X1, X2, and all the data X
#X1 = input_data.A
#X2 = input_data.B
#X3 = input_data.month
#X3=LabelEncoder(input_data.month)
#X4 = input_data.day
#X4=LabelEncoder(input_data.day)
X5 = input_data.INC
X6 = input_data.RACE_bin
X7 = input_data.EDUC
X8=input_data.TRANTIME
X9=input_data.ROOMS
#X10=input_data.RH
#X11=input_data.wind
#X12=input_data.rain
#X13=input_data.area
X = np.column_stack((X5,X6,X7,X8,X9))


#Calculate and show covariance matrix
print("Covariance matrix")
print np.cov(X, rowvar=0).round(2) #rowvar=0 means that each column is a variable. Anything else suggest each row is a variable.
print("")

#Calculate and show correlation coefficients between datasets
print("Correlation Coefficients")
print np.corrcoef(X, rowvar=0).round(2)
print("")

#Define the PCA algorithm
ncompo = 1
print("")
pca = PCA(n_components=ncompo)

#Find the PCA
pcafit = pca.fit(X) #Use all data points since we are trying to figure out which variables are relevant

print("Mean")
print(pcafit.mean_)
print("")
print("Principal Components Results")
print(pcafit.components_)
print("")
print("Percentage variance explained by components")
print(pcafit.explained_variance_ratio_)
print("")

#Plot percentage variance explained by components 
perc = pcafit.explained_variance_ratio_
perc_x = range(1, len(perc)+1)
plt.plot(perc_x, perc,'go--')
plt.xlabel('Components')
plt.ylabel('Percentage of Variance Explained')
plt.title('Varience Explain by Each Component')
plt.savefig(file_name, dpi=300)
plt.show()

'''
#Fun 3D Plot
input_plot = raw_input("3D Plot (Y):")
if input_plot == 'Y' or input_plot == 'y':

    plt.clf()
    le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
    le.fit(input_data.Grade)
    number = le.transform(input_data.Grade)
    colormap = np.array(['blue', 'green', 'orange', 'red'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X4, X5, X6, c=colormap[number])

    ax.set_xlabel('X4')
    ax.set_ylabel('X5')
    ax.set_zlabel('X6')

    plt.show()

'''