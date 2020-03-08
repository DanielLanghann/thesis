# Transporteur ANN

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

# 8 Datenauswertung

cm1     = confusion_matrix(y_test[:,0], y_pred[:,0])
cm2     = confusion_matrix(y_test[:,1], y_pred[:,1])
cm3     = confusion_matrix(y_test[:,2], y_pred[:,2])
cm4     = confusion_matrix(y_test[:,3], y_pred[:,3])
cm5     = confusion_matrix(y_test[:,4], y_pred[:,4])
cm6     = confusion_matrix(y_test[:,5], y_pred[:,5])
cm7     = confusion_matrix(y_test[:,6], y_pred[:,6])
cm8     = confusion_matrix(y_test[:,7], y_pred[:,7])
cm9     = confusion_matrix(y_test[:,8], y_pred[:,8])
cm10    = confusion_matrix(y_test[:,9], y_pred[:,9])
cm11    = confusion_matrix(y_test[:,10], y_pred[:,10])
cm12    = confusion_matrix(y_test[:,11], y_pred[:,11])
cm13    = confusion_matrix(y_test[:,12], y_pred[:,12])
cm14    = confusion_matrix(y_test[:,13], y_pred[:,13])
cm15    = confusion_matrix(y_test[:,14], y_pred[:,14])
cm16    = confusion_matrix(y_test[:,15], y_pred[:,15])

accuracy_rate01 = (cm1[0,0]+cm1[1,1])/2000
accuracy_rate02 = (cm2[0,0]+cm2[1,1])/2000
accuracy_rate03 = (cm3[0,0]+cm3[1,1])/2000
accuracy_rate04 = (cm4[0,0]+cm4[1,1])/2000
accuracy_rate05 = (cm5[0,0]+cm5[1,1])/2000
accuracy_rate06 = (cm6[0,0]+cm6[1,1])/2000
accuracy_rate07 = (cm7[0,0]+cm7[1,1])/2000
accuracy_rate08 = (cm8[0,0]+cm8[1,1])/2000
accuracy_rate09 = (cm9[0,0]+cm9[1,1])/2000
accuracy_rate10 = (cm10[0,0]+cm10[1,1])/2000
accuracy_rate11 = (cm11[0,0]+cm11[1,1])/2000
accuracy_rate12 = (cm12[0,0]+cm12[1,1])/2000
accuracy_rate13 = (cm13[0,0]+cm13[1,1])/2000
accuracy_rate14 = (cm14[0,0]+cm14[1,1])/2000
accuracy_rate15 = (cm15[0,0]+cm15[1,1])/2000
accuracy_rate16 = (cm16[0,0]+cm16[1,1])/2000
