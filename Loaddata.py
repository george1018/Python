import sys
import scipy
import numpy
import matplotlib
import pandas as pd
import sklearn

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = ('/Users/jloh/Desktop/Uni Frankfurt/Qualifikationsphase/WPM/INFO/Python Code/Loaddata/HomeA-meter2_2014.csv')
names =['Date & Time','use [kW]','gen [kW]','FurnaceHRV [kW]','CellarOutlets [kW]','WashingMachine [kW]','FridgeRange [kW]','DisposalDishwasher [kW]','KitchenLights [kW]','BedroomOutlets [kW]','BedroomLights [kW]','MasterOutlets [kW]','MasterLights [kW]','DuctHeaterHRV [kW]']
# lesen aus der CSV Datei
dataset = read_csv(url, names = names)
dataset = dataset.drop([0], axis=0)
# zeigt die Anzahl der Zeile und Spalte an
print ('Zeile und Spalte:', dataset.shape)
# zeigt den Datentyp an
#print (dataset.dtypes)

# zeigt die ersten 50 Zeilen von der Tabelle an
pd.set_option('display.max_columns', None) # alle Spalte anzeigen
pd.set_option('precision', 3) # mit zwei Nachkommastellen
print (dataset.head(50)) # Nur die ersten 50 Zeilen anzeigen

# statistische beschreibung (Mittelwert; Minwert; Maxwert etc.)
#print (dataset.describe())

# Data Visualization

# zeigt ein Histogramm an, wie oft einzelne Werte gemessen wurden
#dataset.hist()
#pyplot.show()

# Korrelation der Werte bestimmen
correlations = dataset.corr(method='pearson')
print (correlations)

# Array bilden
array = dataset.values
X = array[:,1:14]
y = array[:,1:14]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
