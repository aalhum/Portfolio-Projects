import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV

orbits_data = pd.read_csv('orbits.csv')
orbits_data = orbits_data.dropna()  #drop rows with missing values
percent_thresh = 0.75
cutoff = round(percent_thresh*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

train_orbits_data = train_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)
test_orbits_data = test_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)


# TRAINING AND TESTING MODEL
#note: we're using the optimal subset of features

#possible options for features
optimal_subset = ['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']  #list of possible features


#the actual train and test data subsets used
used_train_features = train_orbits_data[optimal_subset]
used_test_features = test_orbits_data[optimal_subset]
train_labels_4 = train_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})
test_labels_4 = test_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})



#Optimize Parameter for Random Forest using cross validation
ranfor_cv = RandomForestClassifier()
list_of_numTrees = [10,20,30,40,50,60,70,80,90,100]
rf_gridsearch = GridSearchCV(
    estimator=ranfor_cv,
    param_grid={'n_estimators':list_of_numTrees}
)

rf_gridsearch.fit(used_train_features,train_labels_4)

st.write("The list of possible number of trees provided to gridsearchCV was:")
st.write(list_of_numTrees)
st.write("The results of using gridsearchCV to search for the optimal number of trees is:")
st.write(rf_gridsearch.best_params_)
#most optimal number of trees, testing 10-100, was 100
#show just one of the actual trees generated using tree.plot_tree
#show the decision boundary generated using Decision_boundary_display.from_estimator which can plot a tree

#tree.plot_tree(rf_pipeline[-1])

ranfor_test = RandomForestClassifier(n_estimators=100)

#fit random forest model using optimal number of trees
ranfor_test.fit(used_train_features,train_labels_4)
accuracy = ranfor_test.score(used_test_features,test_labels_4)
st.write("The accuracy of the random forest model with 100 trees on the test set was:")
st.write(accuracy)
#97.9% accuracy