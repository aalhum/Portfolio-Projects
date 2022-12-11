import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

#IM ON THIS
st.title("K Nearest Neighbors Algorithm")

st.write("K-Nearest-Neighbors algorithm classifies each point based on its closest K neighbors...")

orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
orbits_data = orbits_data.dropna()  #drop rows with missing values
cutoff = round(.75*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

train_orbits_data = train_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)
test_orbits_data = test_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)

#use only an allowed subset of features
optimal_subset = ['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']  #list of possible features
train_subset = train_orbits_data[optimal_subset]
test_subset = test_orbits_data[optimal_subset]
#combine the labels into 4 labels instead of all 7 labels
train_labels_4 = train_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})
test_labels_4 = test_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})


#note: we're using all of the features, not just 2
#NOTE: would be a good idea to show an example KNN changing in an animation using plotly
#maybe I should use the built-in function to find the optimal k-value instead of manually doing it
k_values = [25,50,100,200,300,400,500,600,750]
scores = np.zeros([2,len(k_values)])
i = 0
st.write("First I attempted to find the optimal K value for the KNN model by manually looping through each K value:")
for k_val in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k_val)
    neigh.fit(train_subset,train_labels_4)  #fit the algorithm to the training data
    score = neigh.score(test_subset,test_labels_4)   #apply to test data 
    scores[0,i] = k_val
    scores[1,i] = score
    i = i + 1

scores_data_frame = pd.DataFrame(data=np.transpose(scores),columns = ['K Value','Average Accuracy'])


st.write("Different K values and their associated classification accuracy")
st.dataframe(scores_data_frame)

st.write("Below I try another method, using the GridSearchCV function (which uses cross validation to select optimal parameter values)")
neigh_cv = KNeighborsClassifier()

K_gridsearch = GridSearchCV(
    estimator=neigh_cv,
    param_grid={'n_neighbors':[10,20,30,40,50,100,200,300,400]}
)

