# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"E:\Sentiment Analysis\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

dataset.shape

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

from nltk.stem import LancasterStemmer
corpus1 = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ls = LancasterStemmer()
    review = [ls.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values    

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Training using the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = gnb_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
  

bias = gnb_classifier.score(X_train, y_train)
bias

variance = gnb_classifier.score(X_test,y_test)
variance


# Training using the Random Forest Classifier model using Parameter Tuning on the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = rf_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_rf)
print(ac)
  

bias = rf_classifier.score(X_train, y_train)
bias

variance = rf_classifier.score(X_test,y_test)
variance



# Training using the Random Forest Classifier model using Hyper Parameter Tuning
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=150, max_features = 'sqrt', criterion='entropy')
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = rf_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_rf)
print(ac)
  

bias = rf_classifier.score(X_train, y_train)
bias

variance = rf_classifier.score(X_test,y_test)
variance

#Random forest classifier is bias variance trade-off

# Training using the Decision Tree Classifier model using Parameter Tuning
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dt = dt_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_dt)
print(ac)
  

bias = dt_classifier.score(X_train, y_train)
bias

variance = dt_classifier.score(X_test,y_test)
variance



# Training using the Decision Tree Classifier model using Hyper Parameter Tuning
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(splitter='random', max_features = 'log2')
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dt = dt_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_dt)
print(ac)
  

bias = dt_classifier.score(X_train, y_train)
bias

variance = dt_classifier.score(X_test,y_test)
variance




# Training using the KNN Classifier model using Parameter Tuning
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = knn_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_knn)
print(ac)
  

bias = knn_classifier.score(X_train, y_train)
bias

variance = knn_classifier.score(X_test,y_test)
variance




# Training using the KNN Classifier model using Hpyer Parameter Tuning
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 8, algorithm='ball_tree', weights = 'distance')
knn_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = knn_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_knn)
print(ac)
  

bias = knn_classifier.score(X_train, y_train)
bias

variance = knn_classifier.score(X_test,y_test)
variance




# Training using the SVM Classifier model using Parameter Tuning
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svc = svc_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_svc)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_svc)
print(ac)
  

bias = svc_classifier.score(X_train, y_train)
bias

variance = svc_classifier.score(X_test,y_test)
variance



# Training using the Logistic Regression model using Parameter Tuning
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred_log = log_reg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_log)
print(ac)
  

bias = log_reg.score(X_train, y_train)
bias

variance = log_reg.score(X_test,y_test)
variance




# Training using the XGBoost Classifier model using Parameter Tuning
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred_xgb= xgb.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix :- \n", cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_xgb)
print(ac)
  

bias = xgb.score(X_train, y_train)
bias

variance = xgb.score(X_test,y_test)
variance


#Applying k-fold cross validation on Logistic Regression

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = log_reg, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) 
  

#Applying k-fold cross validation on SVM Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svc_classifier, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) 

#I got maximum accuracy of 77.38% by using K-fold on SVM classifier

#Applying k-fold cross validation on KNN Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn_classifier, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) 


#Applying k-fold cross validation on Decision Tree Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = dt_classifier, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))  



#Applying k-fold cross validation on Random Forest Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rf_classifier, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))  



#Applying k-fold cross validation on Naive Bias Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = gnb_classifier, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) 


#Applying k-fold cross validation on XGBoost Classifier

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X =  X_train, y = y_train, cv = 10 )
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) 


#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}]
grid_search =   GridSearchCV(estimator = svc_classifier,
                             param_grid = parameters,
                             scoring = 'accuracy',
                             cv = 5,
                             n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f}%".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
 
 
#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
forest_params = [{'max_depth': list(range(10,15)), 'max_features': list(range(0,14))}] 

clf = GridSearchCV(rf_classifier, forest_params, cv = 10, scoring='accuracy')
clf.fit(X_train, y_train)
best_parameters = clf.best_params_
best_accuracy = clf.best_score_
print("Best Accuracy : {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)