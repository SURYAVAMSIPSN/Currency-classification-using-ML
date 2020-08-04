import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from matplotlib.pyplot import colorbar

# Useful way of getting data. 
data_set = [line.rstrip('\n').split(',') for line in open('data_banknote_authentication.txt')] # Read text file. 

df = pd.DataFrame(data = data_set) # Convert it to a pandas dataframe. 
df = df.rename(columns={0: "variance", 1: "skewness",2:"kurtosis",3:"entropy",4:'class'}) # rename, for better understanding
df = df.astype(float)
df['class'] = df['class'].astype(int) # make classes of integer type and other features as float 



corr_matrix = df.corr(method = 'kendall')
cov_matrix = df.cov()
print('Correlation:')
print(corr_matrix)

print('\nCovariance:')
print(cov_matrix)
df.plot()
