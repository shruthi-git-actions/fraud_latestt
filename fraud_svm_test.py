import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix


x_test = pd.read_csv('Data/x_test.csv')
x_test = x_test.iloc[1: , :]
y_test = pd.read_csv('Data/y_test.csv')
x_test = x_test.iloc[: , 1:]
y_test=y_test.iloc[: , 1:]
Pkl_Filename = "Models/Fraud_SVM.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    SVM = pickle.load(file)

predictions_SVM = SVM.predict(x_test)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, predictions_SVM)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
import json
with open('Output/Accuracy.json', 'w') as f:
    json.dump((accuracy_score(predictions_SVM, y_test)*100), f)
'''
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.title("AUC(ROC curve)")

plot_confusion_matrix(SVM, x_test, y_test, normalize='true')
plt.show()
'''
a=confusion_matrix(y_test,predictions_SVM)
b=a.tolist()
with open('Output/Confusion_matrix.json', 'w') as f:
    json.dump(b, f)
