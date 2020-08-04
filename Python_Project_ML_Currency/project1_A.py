# SIVANAGA SURYA VAMSI POPURI
# ASU ID: 1217319207
import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
from matplotlib import cm as cm         # for the color map
import seaborn as sns                   # data visualization

################################################################################
# function to create covariance for dataframes                                 #
# Inputs:                                                                      #
#    mydataframe - the data frame to analyze                                   #
#    numtoreport - the number of highly correlated pairs to report             #
# Outputs:                                                                     #
#    correlations are printed to the screen                                    #
################################################################################

def mosthighlycorrelated(mydataframe, numtoreport): 
    cormatrix = mydataframe.corr()                      # find the correlations 

    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 

    # find the top n correlations 
    #print(cormatrix)
    cormatrix = cormatrix.stack()     # rearrange so the reindex will work...
    #print(cormatrix)

    # Reorder the entries so they go from largest at top to smallest at bottom
    # based on absolute value
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    #print(cormatrix)

    # assign human-friendly names 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    print("\nMost Highly Correlated")
    print(cormatrix.head(numtoreport))     # print the top values

################################################################################
# Function to create the Correlation matrix                                    #
# Input:                                                                       #
#    X - a dataframe                                                           #
# Output:                                                                      #
#    The correlation matrix plot                                               #
################################################################################

def correl_matrix(X):
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7,7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet',30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0,len(X.columns),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(X.columns,fontsize=9)
    ax1.set_yticklabels(X.columns,fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()

################################################################################
# Function to create the pair plots                                            #
# Input:                                                                       #
#    df - a dataframe                                                          #
# Output:                                                                      #
#    The pair plots                                                            #
################################################################################

def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')   # set the apearance
    a = sns.pairplot(df,height=2.5)                      # create the pair plots
    a.savefig("pair_plot.jpg")
    plt.show()                                       # and show them

# this creates a dataframe similar to a dictionary
# a data frame can be constructed from a dictionary
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
#iris = pd.read_csv('data_banknote_authentication.txt')
data_set = [line.rstrip('\n').split(',') for line in open('data_banknote_authentication.txt')] # Read text file. 
df = pd.DataFrame(data = data_set)
df = df.astype(float)
df = df.rename(columns={0: "variance", 1: "skewness",2:"kurtosis",3:"entropy",4:'class'}) # rename, for better understanding
df['class'] = df['class'].astype(int)
print('first 5 observations\n',df.head(5))
cols = df.columns

#  descriptive statistics
print('\nDescriptive Statistics')
print(df.describe())

mosthighlycorrelated(df,5)                # generate most highly correlated list
correl_matrix(df)                         # generate the covariance heat plot
pairplotting(df)                          # generate the pair plot

# To print the covariance matrix. 
cov_matrix = df.cov()
print('\nCovariance:')
print(cov_matrix)


