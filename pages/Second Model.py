import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


#neigh = KNeighborsClassifier(n_neighbors=50)  #Set K equal - TRY DIFF K VALUES, perhaps show results for differing values
#of k inside a chart
#neigh.fit(train_orbits_data,train_labels)

#predicted_labels = neigh.predict(test_orbits_data)

#st.write(predicted_labels)  #the labels predicted by the model
#st.dataframe(test_labels.unique())
#st.write(confusion_matrix(test_labels,predicted_labels))


# list of labels: 'Apollo Asteroid','Amor Asteroid (Hazard)','Apollo Asteroid (Hazard)','Aten Asteroid','Aten Asteroid (Hazard)','Amor Asteroid (Hazard)','Apohele Asteroid','Apohele Asteroid (Hazard)'

k_values = [25,50,100,200,300,400,500,600,750]
scores = np.zeros([2,len(k_values)])
i = 0

for k_val in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k_val)
    neigh.fit(train_orbits_data,train_labels)  #fit the algorithm to the training data
    score = neigh.score(test_orbits_data,test_labels)   #apply to test data 
    scores[0,i] = k_val
    scores[1,i] = score
    i = i + 1

scores_data_frame = pd.DataFrame(data=np.transpose(scores),columns = ['K Value','Average Accuracy'])

st.write("Different K values and their associated classification accuracy")
st.dataframe(scores_data_frame)