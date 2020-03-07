# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:27:09 2020

@author: Daniel Langhann
"""

""" ---- Auftragsdaten Random Forest Regression --- """

# Working Directory
import os
os.chdir('C:\\Users\\Daniel Langhann\\OneDrive\\Uni\\Thesis\\data')

# 1 Import Bibliotheken
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime as dt

# 2 Datenimport
dataset = pd.read_csv('auftragsdaten.csv')

# 3 Aufteilung in X,y
X = dataset.iloc[:,0:10].values
y = dataset.iloc[:,10:].values

# 4 Splitting - Training-Set / Test-Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6 Fitting Multiple Random Forest
now1 = dt.now().timestamp()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=150, random_state=0)
regressor.fit(X_train,y_train)
now2 = dt.now().timestamp()
timediff = now2-now1

# 7 Vorhersage der Test-Set-Ergebnisse
y_pred = regressor.predict(X_test)
y_pred = np.round(y_pred,2)

# Negative Werte auf 0 setzen
for i in range(0,len(y_pred)): 
    for j in range(0,4):
        if y_pred[i,j] < 0:
            y_pred[i,j] = 0

# 8 Datenauswertung

""" Labels """

# Artikellabel
myarticlelist = ['Art 1','Art 2','Art 3','Art 4','Art 5',
                 'Art 6','Art 7','Art 8','Art 9','Art 10']

# Labels für Ladehilfsmittel
lhm_labels = ['EP','GB','Karton_gr','Karton_kl']

# Labels für Barcharts setzen
def autolabel(rects, xpos='center'):
    xpos = xpos.lower()  # Parameter normalisieren
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

""" Allgemeine Daten """

# Summe der einzelnen Ladehilfsmittel (Verwendungshäufigkeit) y
lhm_sum_pro_lt = np.sum(y, axis = 0)
# Summenverteilung Ladehilfsmittel als Barchart
plt.rcParams['figure.figsize']=8,4
ind = np.arange(len(lhm_sum_pro_lt))  # x locations
width = 0.35  # Breite der Bars
fig, ax = plt.subplots()
barchart_ltsum = ax.bar(ind, lhm_sum_pro_lt, width, label='Anzahl pro LHM', color = 'royalblue')
# Labeltext, Titel, x-axis tick labels, etc.
ax.set_ylabel('Anzahl')
ax.set_ylabel('LHM')
ax.set_title('Anzahl pro Ladehilfsmittel')
ax.set_xticks(ind)
ax.set_xticklabels(lhm_labels)
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
autolabel(barchart_ltsum)
plt.show()

# Summe der einzelnen Artikel (Verwendungshäufigkeit)
art_sum = np.sum(X, axis = 0)
# Histogram zu Summenabweichung pro Datensatz
sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)
# Plot a simple histogram with binsize determined automatically
ax = sns.distplot(art_sum, kde=False, color="b")
ax.set_ylabel('Häufigkeit %')
ax.set_xlabel('Anzahl Artikel')

# Summe der einzelnen Artikel (Verwendungshäufigkeit) Barchart
plt.rcParams['figure.figsize']=8,4
ind = np.arange(len(art_sum))  # x locations
width = 0.35  # Breite der Bars
fig, ax = plt.subplots()
barchart_artsum = ax.bar(ind, art_sum, width, label='Anzahl pro Artikel', color = 'slategray')
# Labeltext, Titel, x-axis tick labels, etc.
ax.set_ylabel('Anzahl')
ax.set_ylabel('Artikel')
ax.set_title('Anzahl pro Artikel')
ax.set_xticks(ind)
ax.set_xticklabels(myarticlelist)
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
autolabel(barchart_artsum)
plt.show()

# Summe über alle Ladehilfsmittel
lhm_sum_ges = np.sum(lhm_sum_pro_lt)

# Relativer Anteil der verwendeten Ladehilfsmittel im Verhältnis der Gesamtsumme
lhm_rel_anteil = lhm_sum_pro_lt/lhm_sum_ges

# Summen der insgesamt genutzten LHM auf Datensatzebene
lhm_sum_pro_ds = np.sum(y_test, axis = 1)

# Durchschnittliche Anzahl aller LHM auf Datensatzebene y_test
lhm_sum_pro_ds_avg = np.mean(lhm_sum_pro_ds)

# Summe der einzelnen Ladehilfsmittel (Verwendungshäufigkeit) y_test
lhm_sum_pro_lhm_y_test = np.sum(y_test, axis = 0)

""" Auswertung der Testergebnisse"""

# Summe der einzelnen Ladehilfsmittel (Verwendungshäufigkeit) y_pred vs y_test
lhm_sum_pro_lhm_y_pred = np.sum(y_pred, axis = 0).astype(int)
# Barchart Summenvergleich Ladehilfsmittel y_test, y_pred
plt.rcParams['figure.figsize']=8,4
ind = np.arange(len(lhm_sum_pro_lhm_y_test))  # x locations
width = 0.35  # Breite der Bars
fig, ax = plt.subplots()
barchart_lhm_y_pred = ax.bar(ind-0.2,lhm_sum_pro_lhm_y_pred , width, label='y_pred', color = 'royalblue')
barchart_lhm_y_test = ax.bar(ind+0.2, lhm_sum_pro_lhm_y_test, width, label='y_test', color = 'slategray')
# Labeltext, Titel, x-axis tick labels, etc.
ax.set_ylabel('Anzahl')
ax.set_xticks(ind)
ax.set_xticklabels(lhm_labels)
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
autolabel(barchart_lhm_y_test)
autolabel(barchart_lhm_y_pred)
plt.show()

# Abweichung pro Ladehilfsmittel y_test und y_pred
y_delta_lhm = lhm_sum_pro_lhm_y_test-lhm_sum_pro_lhm_y_pred

# Differenz zwischen y_test und y_pred
y_delta = np.array([y_test-y_pred]).reshape(2000,4)
y_delta = np.square(y_delta)
y_delta = np.sqrt(y_delta)

# Summenabweichung pro Datensatz
y_delta_rowsum_vec = np.sum(y_delta,axis=1)
# Histogram zu Summenabweichung pro Datensatz
sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)
# Plot a simple histogram with binsize determined automatically
ax = sns.distplot(y_delta_rowsum_vec, kde=True, color="b")
ax.set_ylabel('Häufigkeit der Abweichungen %')
ax.set_xlabel('Abweichungen')

# Summe aller Datensatz-Abweichungen
y_delta_rowsum_val = np.sum(y_delta_rowsum_vec)

# Durchschnittliche Abweichung aller Summenabweichungen pro Datensatz
y_delta_rowsum_avg = np.mean(y_delta_rowsum_vec)

# Durchschnittliche Abweichung aller Summenabweichungen pro Datensatz 
# bezogen auf die durchschnittliche Anzahl der verwendeten Artikel pro Datensatz
y_delta_rowsum_avg_p = y_delta_rowsum_avg / lhm_sum_pro_ds_avg

# Durchschnittliche Trefferquote:
accuracy_rate = 1-y_delta_rowsum_avg_p

# Durchschnittliche Abweichung pro Zeile von y_delta pro Ladehilfsmittel
y_delta_rowavg = y_delta_rowsum_vec/4

# Maximale Durchschnittliche Abweichung pro Zeile von y_delta pro Ladehilfsmittel
y_delta_rowavg_max = np.max(y_delta_rowavg)

# Minimale Durchschnittliche Abweichung pro Zeile von y_delta pro Ladehilfsmittel
y_delta_rowavg_min = np.min(y_delta_rowavg)

# Median Durchschnittliche Abweichung pro Zeile von y_delta pro Ladehilfsmittel
y_delta_rowavg_median = np.median(y_delta_rowavg)

# Standardabweichung Durchschnittliche Abweichung pro Zeile von y_delta pro Ladehilfsmittel
y_delta_rowavg_std = np.std(y_delta_rowsum_vec)

# Durchschnittliche Abweichung Europalette
y_delta_column_avg_ep = np.mean(y_delta[:,0])

# Durchschnittliche Abweichung Gitterbox
y_delta_column_avg_gb = np.mean(y_delta[:,1])

# Durchschnittliche Abweichung Karton_gross
y_delta_column_avg_kkg = np.mean(y_delta[:,2])

# Durchschnittliche Abweichung Karton_klein
y_delta_column_avg_kkl = np.mean(y_delta[:,3])

y_delta_column_avg_values = np.array([y_delta_column_avg_ep,y_delta_column_avg_gb,
                      y_delta_column_avg_kkg, y_delta_column_avg_kkl])

# Durchschnittliche gewichtete Abweichung
def gewichteteAbweichung(value_list, parameter_list):
    try:
        res = 0
        for i in range(len(value_list)):
            res += value_list[i]*parameter_list[i]
        return res
    except:
        print('Keine Berechnung')

y_delta_column_wavg = gewichteteAbweichung(y_delta_column_avg_values,lhm_rel_anteil)        
