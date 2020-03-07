# Distributions-Art Random Forest

# Working Directory
import os
os.chdir('C:\\Users\\Daniel Langhann\\OneDrive\\Uni\\Thesis\\Data')

# 1 Import Bibliotheken

import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 2 Datenimport

dataset = pd.read_csv('distributions-art.csv')

# 3 Aufteilung in X,y
X = dataset.iloc[:,0:8].values

""" LCL_LKW """

y_lcl_lkw = dataset.iloc[:,8].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train_lcl_lkw, y_test_lcl_lkw = train_test_split(
    X, y_lcl_lkw,
    test_size = 0.20, 
    random_state = 0)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting RF
now1 = dt.now().timestamp()
classifier = RandomForestClassifier(n_estimators=150, random_state = 0)
classifier.fit(X_train, y_train_lcl_lkw)
now2 = dt.now().timestamp()
timediff = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred_lcl_lkw = classifier.predict(X_test)

# 8 Datenauswertung
cm_lcl_lkw = confusion_matrix(y_test_lcl_lkw,y_pred_lcl_lkw)

accuracy_lcl_lkw = (cm_lcl_lkw[0,0]+cm_lcl_lkw[1,1])/2000

""" FCL_LKW """

# 3 Aufteilung in X,yg
y_fcl_lkw = dataset.iloc[:,9].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train_fcl_lkw, y_test_fcl_lkw = train_test_split(
    X, y_fcl_lkw,
    test_size = 0.20,
    random_state = 0)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting RF
now1 = dt.now().timestamp()
classifier = RandomForestClassifier(n_estimators=150, random_state = 0)
classifier.fit(X_train, y_train_fcl_lkw)
now2 = dt.now().timestamp()
timediff2 = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred_fcl_lkw = classifier.predict(X_test)

# 8 Datenauswertung
cm_fcl_lkw = confusion_matrix(y_test_fcl_lkw,y_pred_fcl_lkw)

accuracy_fcl_lkw = (cm_fcl_lkw[0,0]+cm_fcl_lkw[1,1])/2000

""" Luftfracht """

# 3 Aufteilung in X,y
y_luftfracht = dataset.iloc[:,10].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train_luftfracht, y_test_luftfracht = train_test_split(
    X, y_luftfracht, 
    test_size = 0.20, 
    random_state = 0)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting RF
now1 = dt.now().timestamp()
classifier = RandomForestClassifier(n_estimators=150, random_state = 0)
classifier.fit(X_train, y_train_luftfracht)
now2 = dt.now().timestamp()
timediff3 = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred_luftfracht = classifier.predict(X_test)

# 8 Datenauswertung
cm_luftfracht = confusion_matrix(y_test_luftfracht,y_pred_luftfracht)

accuracy_luftfracht = (cm_luftfracht[0,0]+cm_luftfracht[1,1])/2000

""" Seefracht """

# 3 Aufteilung in X,y
y_seefracht = dataset.iloc[:,11].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train_seefracht, y_test_seefracht = train_test_split(
    X, y_seefracht,
    test_size = 0.20, 
    random_state = 0)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting RF
now1 = dt.now().timestamp()
classifier = RandomForestClassifier(n_estimators=150)
classifier.fit(X_train, y_train_seefracht)
now2 = dt.now().timestamp()
timediff4 = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred_seefracht = classifier.predict(X_test)

# 8 Datenauswertung
cm_seefracht = confusion_matrix(y_test_seefracht,y_pred_seefracht)

accuracy_seefracht = (cm_seefracht[0,0]+cm_seefracht[1,1])/2000

# 8 Datenauswertung Gesamt

# Labels
labels = ['LCL_LKW','FCL_LKW','Luftfracht','Seefracht']

# Labels für Barcharts setzen
def autolabel(rects, xpos='center'):
    xpos = xpos.lower()  # Parameter normalisieren
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

# Summe pro Distributions-Art
sum_lcl_lkw     = np.sum(y_lcl_lkw)
sum_fcl_lkw     = np.sum(y_fcl_lkw)
sum_luftfracht  = np.sum(y_luftfracht)
sum_seefracht   = np.sum(y_seefracht)
alle = [sum_lcl_lkw,sum_fcl_lkw,sum_luftfracht,sum_seefracht]

# Verteilung Distributions-Art
plt.rcParams['figure.figsize']=8,4
ind = np.arange(len(alle))  # x locations
width = 0.35  # Breite der Bars
fig, ax = plt.subplots()
barchart_dk = ax.bar(ind, alle, width, label='Häufigkeit pro Distributions-Art', color = 'royalblue')
ax.set_ylabel('Häufigkeit')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
autolabel(barchart_dk)
plt.show()

# Accuracy-Werte
p = 100
alle_accuracy = [accuracy_lcl_lkw, accuracy_fcl_lkw, accuracy_luftfracht, accuracy_seefracht]
alle_accuracy = p*np.array(alle_accuracy)
# Accuracy Rate
plt.rcParams['figure.figsize']=8,4
ind = np.arange(len(alle_accuracy))  # x locations
width = 0.35  # Breite der Bars
fig, ax = plt.subplots()
barchart_dk = ax.bar(ind, alle_accuracy, width, label='Accuracy-Rate % pro Distributions-Art', color = 'royalblue')
ax.set_ylabel('%')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
autolabel(barchart_dk)
plt.show()
