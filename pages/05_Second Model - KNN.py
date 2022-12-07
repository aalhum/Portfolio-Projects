import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#IM ON THIS
st.title("K Nearest Neighbors Algorithm")

st.write("K-Nearest-Neighbors algorithm classifies each point based on its closest K neighbors...")

orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
orbits_data = orbits_data.dropna()  #drop rows with missing values
cutoff = round(.9*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

train_orbits_data = train_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)
test_orbits_data = test_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)

#use only an allowed subset of features
allowed_subset = ['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']  #list of possible features
train_subset = train_orbits_data[allowed_subset]
test_subset = test_orbits_data[allowed_subset]
#combine the labels into 4 labels instead of all 7 labels
train_labels_4 = train_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})
test_labels_4 = test_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})


#note: we're using all of the features, not just 2
#NOTE: would be a good idea to show an example KNN changing in an animation using plotly
#maybe I should use the built-in cross validation to find the optimal k-value instead of manually doing it
k_values = [25,50,100,200,300,400,500,600,750]
scores = np.zeros([2,len(k_values)])
i = 0

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

#get the k-value with the highest accuracy
optimal_k_value = max(scores[1,:])
st.write('The k-value with the highest accuracy on the test set is: ')
st.write(optimal_k_value)
#The optimal value for K based on the cross validation seems to be 50, with the highest accuracy
neigh_opt = KNeighborsClassifier(n_neighbors=optimal_k_value)
neigh_opt.fit(train_subset,train_labels_4)
accuracy_opt = neigh.score(test_subset,test_labels_4)
st.write("The accuracy of the model (with optimal K-value) on the test set is:")
st.write(accuracy_opt)





#RUNNING THE ALGORITHM FOR REAL, WITH CHOSEN FEATURES - JUST REPLACE THE FEATURES BELOW
#allowed_train
#allowed_test

st.write('When we use all of the features available to predict 4 asteroid classes, the resulting accurate is:')
b_pipeline = Pipeline([('ala_scaler2',StandardScaler()),('ala_svc2',svm.SVC(kernel='rbf'))])
#fit the model
b_pipeline.fit(allowed_train,train_labels_4)

#predict on test set
accuracy_full = b_pipeline.score(allowed_test,test_labels_4)

st.write("The accuracy of the SVM with linear kernel on the test data is:")
st.write(accuracy_full)
#90.23% accuracy with rbf kernel
