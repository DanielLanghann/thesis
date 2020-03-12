# Transporteur ANN

# Working Directory
# import os
# os.chdir('C:\\Users\\Daniel Langhann\\OneDrive\\Uni\\Thesis\\Data')

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

# 2 Datenimport
dataset = pd.read_csv('serviceprovider.csv')

# 3 Aufteilung in X,y
X = dataset.iloc[:,0:12].values
y = dataset.iloc[:,12:].values

# 4 Splitting - Training-Set / Test-Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# 5 Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting ANN
classifier = Sequential()
# Add Input Layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
# Add Hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
# Add Output layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling ANN
# Optimizer = adam -> Stochastic Gradient Descent (Ermittlung der optimalen Gewichte)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting ANN
now1 = dt.now().timestamp()
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
now2 = dt.now().timestamp()
timediff = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
