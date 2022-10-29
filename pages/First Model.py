import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#citation: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

#CODE FOR FIRST MODEL - SVM

st.title('Support Vector Machine')

st.write("A Support Vector Machine (SVM) classifies data points into different classes by calculating a decision boundary, a separation between the different groups.")


orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
#st.write(labels.unique())    #check how many unique values are in asteroid classification
orbits_data = orbits_data.drop(['Object Name','Orbital Reference'],axis=1) #drop columns not used as features
orbits_data = orbits_data.dropna()  #drop rows with missing values
cutoff = round(.9*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

#display the training and testing data using containers
train_container = st.container()
test_container = st.container()
traindataexpander1 = st.expander("Open Training Data")
traindataexpander2 = st.expander("Open Training Labels")
testdataexpander1 = st.expander("Open Testing Data")
testdataexpander2 = st.expander("Open Test Labels")
traincol1, traincol2 = st.columns(2, gap='medium')  
testcol1, testcol2 = st.columns(2, gap='large')

with train_container:
    with traincol1:
        with traindataexpander1:
            st.dataframe(train_orbits_data)
    with traincol2:
        with traindataexpander2:
            st.dataframe(train_labels)
        
with test_container:
    with testcol1:
        with testdataexpander1:
            st.dataframe(test_orbits_data)
    with testcol2:
        with testdataexpander2:
            st.dataframe(test_labels)




def assign_labels (labs):  #change classification labels for asteroid classification to numerical values 
    for i in range(0,labs.size):
        if labs.iloc[i] == 'Apollo Asteroid':
            labs.iloc[i] = 0
        elif labs.iloc[i] == 'Amor Asteroid (Hazard)':
            labs.iloc[i] = 1
        elif labs.iloc[i] == 'Apollo Asteroid (Hazard)':
            labs.iloc[i] = 2
        elif labs.iloc[i] == 'Aten Asteroid':
            labs.iloc[i] = 3
        elif labs.iloc[i] == 'Aten Asteroid (Hazard)':
            labs.iloc[i] = 4
        elif labs.iloc[i] == 'Amor Asteroid (Hazard)':
            labs.iloc[i] = 5
        elif labs.iloc[i] == 'Apohele Asteroid':
            labs.iloc[i] = 6
        elif labs.iloc[i] == 'Apohele Asteroid (Hazard)':
            labs.iloc[i] = 7
    return labs



clf = svm.SVC()
clf.fit(train_orbits_data,train_labels)

st.write("Overall Distribution of Asteroid Classifications, Training + Testing Data")
st.write(orbits_data['Object Classification'].value_counts())



predictions = clf.predict(test_orbits_data)
accuracy = clf.score(test_orbits_data,test_labels)
st.write(np.unique(predictions))
confusion_matrix(test_labels,predictions)

#ConfusionMatrixDisplay.from_estimator(clf,test_labels,predictions)
