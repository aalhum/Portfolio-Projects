import streamlit as st
import numpy as np
import statistics
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
from sklearn.feature_selection import RFECV, SelectKBest, chi2

st.title("Feature Selection")

orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
#st.write(labels.unique())    #check how many unique values are in asteroid classification
orbits_data = orbits_data.drop(['Object Name','Orbital Reference'],axis=1) #drop columns not used as features
orbits_data = orbits_data.dropna()  #drop rows with missing values
cratio = 0.75  #percentage of data used for training
cutoff = round(cratio*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')



#let's try to reduce number of features using recursive feature elimination
#using cross validation to find the optimal number of features:
#poss_min_features_to_select = 2
#an_svc = svm.SVC(kernel='linear')
#only allowing features that are not directly used for the definitions of the asteroid classifications or whether they are considered hazardous or not
#st.write("Given below are the features that we considered. We excluded features that are included in the definition of the different asteroid classes and the definition of \'hazardous\'")
#an_svc.fit(allowed_features,train_labels)
#rfecv = RFECV(
#    estimator=an_svc,
#    step=1,
#    cv=StratifiedKFold(2),
#    scoring="accuracy",
#    min_features_to_select=poss_min_features_to_select,
#    )
#rfecv.fit(allowed_features,test_labels)
    
#st.write("Optimal number of features: %d" % rfecv.n_features_)
    
#fig_opt, ax_opt = plt.subplots()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (accuracy)")
#plt.plot(
#    range(poss_min_features_to_select, len(rfecv.grid_scores_) + poss_min_features_to_select),
#    rfecv.grid_scores_,
#       )
    
#st.pyplot(fig_opt)


#use only an allowed subset of all of the features, create the training set
allowed_features = train_orbits_data[['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']]
new_labels = train_labels.copy()
quad_labels = new_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})


min_features_to_select = 1
feature_scaler = StandardScaler()
#use StandardScaler function to Scale the data to have 0 mean and standard deviation of 1
standardized_train = feature_scaler.fit_transform(allowed_features)

feat_model = svm.SVC(kernel='linear')
#train the SVM model to use with RFECV
feat_model.fit(standardized_train,quad_labels)
#define cross validation fold scheme
cv = StratifiedKFold(5)  

#DEFINE RECURSIVE FEATURE SELECTION WITH CROSS VALIDATION
ala_rfecv = RFECV(
    estimator=feat_model,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2
)

ala_rfecv.fit(standardized_train,quad_labels)

st.write("We ran the function RFECV, which does recursive feature elimination using a cross validation scheme to find the optimal number of features") 
st.write("We used a linear SVM in order to score the features, and we used a 5-fold (stratified) cross validation scheme to find the optimal number of features")
st.write(f"Optimal number of features: {ala_rfecv.n_features_}")
st.write(f"out of a total number of {ala_rfecv.n_features_in_} features")

st.write("Thus including all of the features is the optimal set, which makes sense")

st.write("When the percentage of training data used was increased from 20% to 50% to 75% of the total dataset, the number of optimal features also increased. This is likely because the number of samples increased so the ratios of each output class were different.")
st.write("The labels for these features were:")
#st.write()

#feat_pipeline = Pipeline([('ala_feat_scaler',StandardScaler()),('ala_feat_svc',svm.SVC(kernel='linear'))])
#feat_pipeline.fit(allowed_features,quad_labels)
#cv = StratifiedKFold(5)

#rfecv = RFECV(
#    estimator=feat_pipeline,
#    step=1,
#    cv=cv,
#    scoring="accuracy",
#    min_features_to_select=min_features_to_select,
#    n_jobs=2,
#)

    
#rfecv.fit(allowed_features,quad_labels)

#st.write(f"Optimal number of features: {rfecv.n_features_}")

