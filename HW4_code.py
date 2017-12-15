'''
CME594 Introduction to Data Science
Homework 4 Code - Introduction to Data Mining
(c) Sybil Derrible
'''

#Libraries needed to run the tool
import numpy as np
import pandas as pd

#Ask for file name
#file_name = raw_input("Name of file:")
file_name = "HW4_data"

#Read the csv file accounting for two-row header and three-column index values
input_data = pd.read_csv(file_name + '.csv', header=[0,1], index_col=[0,1,2])
#data = input_data.apply(pd.to_numeric, errors='coerce')
data = input_data

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

with np.errstate(divide='ignore', invalid='ignore'): #ignore divide errors when divide by 0.
    mode_car = (data.HC02_EST_VC01.values.astype(float) + data.HC03_EST_VC01.values) / data.HC01_EST_VC01.values
    mode_pt = data.HC04_EST_VC01.values.astype(float) / data.HC01_EST_VC01.values
    
    

with np.errstate(divide='ignore', invalid='ignore'): #ignore divide errors when divide by 0.
    mode_sf=data.HC04_MOE_VC03.values.astype(float) #Define one as float to define all as float


with np.errstate(divide='ignore', invalid='ignore'): #ignore divide errors when divide by 0.
    mode_pt_mbsao=data.HC04_EST_VC38.values.astype(float) #Define one as float to define all as float

print("Number of times both a: {0}".format((mode_sf>10).sum()))
print("Number of times b > 2: {0}".format((mode_pt_mbsao>15).sum()))
print("Number of times both a and b > 2: {0}".format(((mode_sf>40)*(mode_pt_mbsao>40)).sum()))

'''

    print("Average car mode share: {0}%".format(100*np.nanmean(mode_car).round(4)))
    print("Median car mode share: {0}%".format(100*np.nanmedian(mode_car).round(4)))
    print("")
    print("Average public transit mode share: {0}%".format(100*np.nanmean(mode_pt).round(4)))
    print("Median public transit mode share: {0}%".format(100*np.nanmedian(mode_pt).round(4)))
    print("")
'''