

Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Now lets load the dataset into pandas dataframe
credit_card_data = pd.read_csv('/content/creditcard.csv')

#First 5 rows of the dataset
#1st column is time of transactions present in seconds (time elapsed in seconds from the first transaction has happened)
# v1 v2 v3.. are the features about particular transaction we cannot have whole credit card details as they are very sensitive hence cannot expose then hence the details are converted by principle component analysis method  and it converts all the features into numerical values which we are going to use

# Last 2nd column shows amount in US dollars
# Class here means it tells if transaction is legit or fraudulent 0/1 where 0 -> legit and 1 -> fraudulent 
# Data is showing transaction of 48 hrs 
credit_card_data.head()

credit_card_data.tail()

#dataset information
credit_card_data.info()

#checking for missing values
credit_card_data.isnull().sum()
#we donot have any missing values else we need process to convert missing values

#distribution of legit transactions and fraudulent transactions 0->normal 1->fraud
#we have very less fraudulent data sets hence we cannot feed to machinle learning so if we
#give new data set it will consider it as normal transaction as more than 99% of data set in one particular class i.e normal transaction
#Hence the process of handling unbalanced dataset comes to play
credit_card_data['Class'].value_counts()

#This dataset is highly unbalanced

#Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#statistical measures of data
#its 25 percentile
legit.Amount.describe()

fraud.Amount.describe()

# The mean value we get here in fraud is higher than than of Legit keep in mind 
# The maximum value we get in fraud is less than max of the legit

#Compare the values for both of the transactions
credit_card_data.groupby('Class').mean()

#Wide difference between fraud and normal transaction, so by this way it can predict legit and fraud transaction
#Dealing with unbalanced data i.e 492 and 2 lakh by undersampling
#In undersampling we are going to build the sample dataset containing similar distribution of normal transaction and fraudent transaction
#Take 492 normal transaction and 492 fraud datas
#Take random 492 values from 2 lakh values -> normal and 492 values -> legit


legit_sample = legit.sample(n=492)
#Lets concatinate the 2 dataframes

new_dataset = pd.concat([legit_sample, fraud], axis=0)
#axis 0 means all legit sample gets added below the

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#Splitting the data into feature and target

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
#previously we have 31 columns now we don't have the class column

print(X)

print(Y)

#Split data into training data & testing data
#by train test function which we have imported
#0.2 means 20% goes to testing data where as 80% data goes to training data and 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#features present in X and labels present in y, splitt x and y in training and testing data splitted randomly. These features will be splitted into training data
#and corresponding labels will be choosen from y. So all the features of training data stored in X train and all the labes for corresponding data stores in Y_test
#Y_test contain all the values of 30 columns and corresponding label obtained from y.
#80% data in X_train and corresponding labels in Y_train and 20% data in X_test and corresponding labels in Y_test, Hence 4 variables and labels  

print(X.shape, X_train.shape, X_test.shape)

#Model Training
#Logistic Regression 

model =LogisticRegression()

#training logistic regression model with training data
model.fit(X_train, Y_train)

#Model evaluation
#Accuracy score

#Accuracy on training data
# predicting labels for x_trains all the labels will be stored in x_train_prediction 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# we compare the values stored in ytrain
# compare values and give accuracy score

print('Accuracy on training data: ', training_data_accuracy)

#accuracy on test data

#y_test is real data for our labels for data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#compare here

print('Accuracy score on test data: ', test_data_accuracy)
