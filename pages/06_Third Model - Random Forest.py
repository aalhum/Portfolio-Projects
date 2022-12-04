import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

orbits_data = pd.read_csv('orbits.csv')
orbits_data = orbits_data.dropna()  #drop rows with missing values
percent_thresh = 0.7
cutoff = round(percent_thresh*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

train_orbits_data = train_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)
test_orbits_data = test_orbits_data.drop(['Object Name','Orbital Reference'],axis=1)

optimal_features = ['Perihelion Argument (deg)','Mean Anomoly (deg)','Orbital Period (yr)']

rf_pipeline = Pipeline([('ala_scaler',StandardScaler()),('ala_rfc',RandomForestClassifier(n_estimators=10))])


# TRAINING AND TESTING MODEL
#note: we're using the optimal subset of features
small_train = train_orbits_data[optimal_features]
small_test = test_orbits_data[optimal_features]

rf_pipeline.fit(small_train,train_labels)  #TRAIN MODEL

#SHOW THE RANDOM FOREST DECISION TREE LABELS - USE ALL OF THE FEATURES AND SEE WHAT THE TOP ONES ARE, THE MOST IMPORTANT ONES.
#COMPARE TO THE OPTIMAL SUBSET YOU FOUND BEFORE

st.write("5-fold cross validation was computed on the training data. The accuracy ratio for each of the 5 folds is given below.")
accuracies = cross_val_score(rf_pipeline,small_train,train_labels, cv=5)
#accuracy = clf.score(test_subset,test_labels)       #OBTAIN ACCURACY
st.write(accuracies)
#st.write(np.unique(predictions))

predictions = rf_pipeline.predict(small_test)   #PREDICT TESTING DATA
accuracy = rf_pipeline.score(small_test,test_labels)
st.write("The test set accuracy is:")
st.write(accuracy)


