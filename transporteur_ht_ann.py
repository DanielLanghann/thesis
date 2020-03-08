# Transporteur HT ANN

# Working Directory
import os
os.chdir('C:\\Users\\Daniel Langhann\\OneDrive\\Uni\\Thesis\\Data')

# 1 Import Bibliotheken
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns


# 2 Datenimport
dataset = pd.read_csv('transporteur_ht.csv')

# 3 Aufteilung in X,y
X = dataset.iloc[:,0:68].values
y = dataset.iloc[:,68:].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting ANN
# ANN initialisieren
classifier = Sequential()
# Add Input Layer
classifier.add(Dense(units = 143, kernel_initializer = 'uniform', activation = 'relu', input_dim = 68))
# Add Hidden layer
classifier.add(Dense(units = 143, kernel_initializer= 'uniform', activation = 'relu'))
# Add Output layer
classifier.add(Dense(units = 143, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting ANN
now1 = dt.now().timestamp()
classifier.fit(X_train, y_train, batch_size = 10, epochs = 15)
now2 = dt.now().timestamp()
timediff = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)

# 8 Datenauswertung

df = pd.DataFrame(y_test)
df.hist()

# Auftraege pro Fahrzeug
auftraege = np.sum(y,axis=0)

sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)
# Plot a simple histogram with binsize determined automatically
ax = sns.distplot(auftraege, kde=False, color="b")
ax.set_xlabel('Aufträge pro Fahrzeug')


cm1     = confusion_matrix(y_test[:,0], y_pred[:,0])

accuracy = np.array([0.9926, 0.9943, 0.9948,0.9950,0.9952,0.9953,0.9954,0.9955,
 0.9955,0.9955,0.9956,0.9956,0.9957,0.9957,0.9957,0.9958,
 0.9958,0.9958,0.9959,0.9958])

accuracy.size
x = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

plt.plot(accuracy, c='royalblue', ls = '--', marker = 's', ms = 7, label="Accuracy-Rate")
plt.legend(loc = 'upper_left', bbox_to_anchor=(1,1))
plt.xticks(list(range(0,20)),x, rotation = 'horizontal')
plt.xlabel("Anzahl Epochs")
plt.ylabel("%")  
plt.show()
