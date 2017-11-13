import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

############### Loading data #####################
TrainingData = pd.read_csv('train.csv')
Final_test_set = pd.read_csv('test.csv')

############## Preprocessing data ####################
X = TrainingData
#Removing columns of the training data which we have not yet found a way to convert into meaningful numerical expressions. 
X = X.drop('Name',1)
X = X.drop('Ticket',1)
X = X.drop('Cabin',1)
X = X.drop('Embarked',1)
#Remove rows(passengers) with NaN's. 
X = X.dropna(axis=0, how='any')
#Extracting the y-column with the list of who survived
y = X['Survived']
#Removing the column of those who survived from X. 
X = X.drop('Survived',1)
#replacing male = 1 and female = 0.
X = X.replace(to_replace = 'male',value= 1)
X = X.replace(to_replace = 'female',value = 0) 
#split the TrainingData into a test and training set
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

############### Data Analysis #####################
#Decision tree classfier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
#clf.predict(X_test)
print('Decision Tree Classfier score',clf.score(X_test,y_test))
#Support Vector Machine classfier
clf = svm.SVC()
clf.fit(X_train,y_train)
print('Support Vector Machine classifier score',clf.score(X_test,y_test))


############################### out commented code ####################################
#Overview of the data: 
#print(TrainingData.shape)
#print(type(TrainingData))
#print(TrainingData.keys())
#print(train.head())

#print(X.head)
#print(X.head())
#X_train = X_train.drop('')

#list of the attributes of pd:
#print(dir(pd))
#Print version of pd:
#print(pd.__version__)

#print(TrainingData['Cabin'])
#print(type(TrainingData))
#print(TrainingData.shape)
#print(TrainingData.keys())
#print(TrainingData)

#Neural Network classfier
#alpha = 0.5
#hidden_nodes = [3,5]
#mlp = MLPClassifier(random_state=0,activation='relu',hidden_layer_sizes=hidden_nodes,alpha = alpha)
#mlp.fit(X_train, y_train)


