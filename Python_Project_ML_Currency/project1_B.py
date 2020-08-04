import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import pickle


# Useful way of getting data. 
data_set = [line.rstrip('\n').split(',') for line in open('data_banknote_authentication.txt')] # Read text file. 

df = pd.DataFrame(data = data_set) # Convert it to a pandas dataframe. 
df = df.rename(columns={0: "variance", 1: "skewness",2:"kurtosis",3:"entropy",4:'class'}) # rename, for better understanding
df = df.astype(float)
df['class'] = df['class'].astype(int)

# From the dataset, the entropy can be dropped, since it has little impact on the prediction. 
# (Low correlation and co-variance with the output classes)
X = np.array(df.drop(columns = ['class'])) # Drop the labels. Take only the the features. 
y = np.array(df['class']) # Take only the labels. 

# Split randomly to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42) 



# Logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print('Logistic regression score: ',clf.score(X_test, y_test)) # Get score against testing data. 

# DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0).fit(X_train,y_train)
print('Decision tree score: ', dtc.score(X_test, y_test))

# Perceptron classifier
pct = Perceptron(tol=1e-3, random_state=0).fit(X_train,y_train)
print('Perceptron score: ', pct.score(X_test, y_test))

# RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train,y_train)
print('Random Forest score: ', rfc.score(X_test, y_test))

# Support vector machine
sv = SVC(gamma='auto').fit(X_train,y_train)
print('SVM score: ', sv.score(X_test, y_test))

# k nearest neighbors
k = 3
nbr = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
print('KNN score: ', nbr.score(X_test, y_test))

# Variance plays important role. Dropping variance --> scores (accuracy) drops significantly.
# Kurtosis plays okayish role. Dropping this --> okayish decrease in score.  
# Skewness --> more or less same effect as dropping variance. 


v = ['Logistic Reg','Dec. Tree','Perceptron','Rand. Forest', 'SVM','KNN'] 
scores = [clf.score(X_test, y_test),dtc.score(X_test, y_test),pct.score(X_test, y_test),
          rfc.score(X_test, y_test),sv.score(X_test, y_test),nbr.score(X_test, y_test)]

# 1,-1 ==> row. 

# Plotting the accuracy scores of various learning models. 
plt.plot(v, scores,'*')
plt.xlabel('Different algorithms used')
plt.ylabel('Accuracy scores')