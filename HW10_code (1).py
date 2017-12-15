'''
CME594 Introduction to Data Science
Homework 10 Code - Network Science
Marina Corby
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


file_name = 'HW10_adj'


#Create a pandas dataframe from the csv file.      
#note that we added index_col=0 since we have row headers too.
input_data = pd.read_csv(file_name + '.csv', header=0, index_col=0) 


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(input_data.index), len(input_data.columns.values)))

#Define graph
g = nx.from_numpy_matrix(input_data.values)

#Print individual properties
#print("#1 way to know number of nodes: {0}".format(g.number_of_nodes())) #two ways to write it
#print("Number of nodes: {0}".format(nx.number_of_nodes(g)))
#print("")

#Print main properties of graph
#print nx.info(g)


#2(a)
#Calculate cyclomatic number, betti number, planar degree of connectivity 

nodes = float(nx.number_of_nodes(g))
links = float(nx.number_of_edges(g))
betti_number = links/nodes
mu = links - nodes + 1 #cyclomatic number
gamma = links/(3*nodes - 6) #degree of connectivity

print("# of nodes is: {0}".format(nodes))
print("# of links is: {0}".format(links))
print("Cyclomatic number is: {0}".format(mu))
print("Betti number is: {0}".format(betti_number))
print("Degree of connectivity is :{0}".format(gamma))



#2(b)
#Calculate degree centrality and betweenness centrality and export as csv
'''
deg_centrality = pd.DataFrame.from_dict(nx.degree_centrality(g), orient = 'index')
deg_centrality.rename(columns={0:'degree_centrality'}, inplace=True)

between_centrality = pd.DataFrame.from_dict(nx.betweenness_centrality(g), orient = 'index')
between_centrality.rename(columns={0:'betweenness_centrality'}, inplace=True)

data_frame= pd.concat([deg_centrality, between_centrality], axis = 1)
print data_frame

data_frame.to_csv("degree_betweenness_centrality.csv")
'''


#2(c)
#Plot graph
#circular, spectral, random, springdf

#define an empty figure
fig = plt.figure()

fig1 = fig.add_subplot(2, 2, 1)
fig1.set_title("Circular")
nx.draw_circular(g, with_labels = True)
plt.draw()

fig2 = fig.add_subplot(2,2,2)
fig2.set_title("Random")
nx.draw_random(g, with_labels = True)
plt.draw()

fig3 = fig.add_subplot(2,2,3)
fig3.set_title("Spectral")
nx.draw_spectral(g, with_labels = True)
plt.draw()


fig4 = fig.add_subplot(2,2,4)
fig4.set_title("Spring")
nx.draw_spring(g, with_labels = True)
plt.draw()
#plt.savefig(file_name + '_plot.png') #Saving the plot
plt.show()

