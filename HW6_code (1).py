'''
CME594 Introduction to Data Science
Homework 6 Code - Clustering Analysis
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage #for dendrogram specifically
import matplotlib.pyplot as plt

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'usa_00006'
input_data = pd.read_csv(file_name + '.csv', header=0, nrows=1000)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns)))
print("")


#Defining X1, X2, and all the data X
X1 = input_data.EDUC.values
X2 = input_data.INC.values


X = np.column_stack((X1, X2))

#Define number of clusters
clusters = 7


#Define which KMeans algorithm to use and fit it
Y_Kmeans = KMeans(n_clusters = clusters)
Y_Kmeans.fit(X)
Y_Kmeans_labels = Y_Kmeans.labels_
Y_Kmeans_silhouette = metrics.silhouette_score(X, Y_Kmeans_labels, metric='sqeuclidean')
print("Silhouette for Kmeans: {0}".format(Y_Kmeans_silhouette))
print("Results for Kmeans: {0}".format(Y_Kmeans_labels))


#Define which hierarchical clustering algorithm to use and fit it
linkage_types = ['ward', 'average', 'complete']
Y_hierarchy = AgglomerativeClustering(linkage=linkage_types[2], n_clusters=clusters)
Y_hierarchy.fit(X)
Y_hierarchy_labels = Y_hierarchy.labels_
Y_hierarchy_silhouette = metrics.silhouette_score(X, Y_hierarchy_labels, metric='sqeuclidean')
print("Silhouette for Hierarchical Clustering: {0}".format(Y_hierarchy_silhouette))
print("Hierarchical Clustering: {0}".format(Y_hierarchy_labels))


#Define figure
colormap = np.array(['cyan', 'black', 'magenta', 'red', 'orange', 'green', 'brown', 'yellow', 'blue', 'white']) #Define colors to use in graph - could use c=Y but colors are too similar when only 2-3 clusters
fig = plt.figure() #Define an empty figure
fig.set_size_inches(8,4) #Define the size of the figure as 8 inches by 4 inches


#Plot KMeans results
fig1 = fig.add_subplot(1,2,1)
plt.title("KMeans")
plt.scatter(X[:, 0], X[:, 1], c=colormap[Y_Kmeans_labels])
plt.annotate("s = " + str(Y_Kmeans_silhouette.round(2)), xy=(1, 0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')

#Plot Hierarchical clustering results
fig1 = fig.add_subplot(1,2,2)
plt.title("Hierarchical Clustering")
plt.scatter(X[:, 0], X[:, 1], c=colormap[Y_hierarchy_labels])
plt.annotate("s = " + str(Y_hierarchy_silhouette.round(2)), xy=(1, 0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')


labels1=['clusters: 4', 'clusters: 7', 'clusters: 10']
cnt=0
plt.figure(2)
for i in range(1,4):
    for j in range(1,4):
        cnt=cnt+1
        plt.subplot(3,3,cnt)
        Y_hierarchy = AgglomerativeClustering(linkage=linkage_types[i-1], n_clusters=3*j+1)
        Y_hierarchy.fit(X)
        Y_hierarchy_labels = Y_hierarchy.labels_
        Y_hierarchy_silhouette = metrics.silhouette_score(X, Y_hierarchy_labels, metric='sqeuclidean')
        if i==1:
            plt.title(labels1[j-1])
        if j==1:
            plt.ylabel(linkage_types[i-1])
        plt.scatter(X[:, 0], X[:, 1], c=colormap[Y_hierarchy_labels])
        plt.annotate("s = " + str(Y_hierarchy_silhouette.round(2)), xy=(1, 0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')
plt.tight_layout()
plt.show()

'''
#Show plots
fig.savefig(file_name + '_clustering.png', dpi=300)
plt.show()

plt.clf() #To erase the current figure for the figure to come

label=input_data.stationame.values   

#Using Scipy to draw dendrograms - for more info, see: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
linkage_types = ['ward', 'average', 'complete']
Z = linkage(X, linkage_types[2])
dendro = plt.figure()
dendro.set_size_inches(18,12)
dendrogram(Z, labels=label,leaf_rotation=90, leaf_font_size=11)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index from Dataframe')
plt.ylabel('Distance')
plt.savefig(file_name + '_dendro.png', dpi=300)
plt.show()


'''



