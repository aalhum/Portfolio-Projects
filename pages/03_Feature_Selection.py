import streamlit as st
import numpy as np
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


#using chi-squared score to score the features (univariate statistical test) and determine which features are more essential for predicting asteroid classification
allowed_features = train_orbits_data[['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']]

optimal_features = SelectKBest(chi2, k=3).fit(allowed_features,train_labels)

which_features = optimal_features.get_support()

allowed_expand = st.expander('Show Initial Feature Set (Before Feature Selection)')
selected_expand = st.expander('Show Selected Features')

st.write("Here is the list of features from Orbital_Characteristics, though some were removed for the following reasons:")
st.write("We removed Object Name and Orbital Reference because they are identifiers")
st.write("We removed Epoch because this is the starting reference time point and therefore not particularly useful.")
st.write("We removed Perihelion Distance, Aphelion Distance, Minimum Orbit Intersection Distance, and Asteroid Magnitude because they are used in the definitions of the asteroid classifications and so it did not make sense to use them for prediction.")

st.dataframe(allowed_features)

st.write("Initially we tried performing feature selection by looking at subsets of features and running cross-validation, but this proved to take an inordinate amount of time, even when we reduced the number of samples used drastically.")
st.write("Therefore we decided to simply use a univariate feature selection (specifically the SelectKBest function in Scikit Learn), where we rank features using a statistical test (ANOVA, Chi-Squared, etc) and then select the top K features.")
st.write("In our case we arbitrarily chose K=3 for the number of optimal features.")

st.write('Below we show which features from the original set were selected.')
st.write(allowed_features.columns[which_features])   #use the indices to get the names of the selected features
    
    