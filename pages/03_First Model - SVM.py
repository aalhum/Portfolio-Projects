import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
#citation: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

#CODE FOR FIRST MODEL - SVM

st.title('Support Vector Machine')

st.write("A Support Vector Machine (SVM) is an algorithm that is best for linearly separable data. It can be used for both classification and regression. It works by selecting \"support vectors\", which are a subset of the data points chosen to represent their classes. The algorithm separates the data points into classes by dividing them using a decision boundary, with the largest possible margin on either side. The margin boundaries are on the support vectors. ")

#EXAMPLE SVC PLOT
#this code + the decision boundary plotting code comes from the scikit site
example_pipe = Pipeline([('example_scaler',StandardScaler()),('example_svc',svm.SVC(kernel='linear'))])
example_pts, example_labels = make_blobs(n_samples = 100,n_features = 2,centers = 2)
example_pipe.fit(example_pts,example_labels)


fig_ex,ax_ex = plt.subplots()

plt.scatter(example_pts[:,0],example_pts[:,1],c=example_labels, cmap=plt.cm.Paired)

DecisionBoundaryDisplay.from_estimator(
    example_pipe,
    example_pts,
    colors="k",
    levels=[-1,0,1],
    alpha=0.5,
    linestyles=["--","-","--"],
    ax=ax_ex,
)

plt.show()
st.pyplot(fig_ex)



#SET UP TRAINING AND TESTING DATA
orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
#st.write(labels.unique())    #check how many unique values are in asteroid classification
orbits_data = orbits_data.drop(['Object Name','Orbital Reference'],axis=1) #drop columns not used as features
orbits_data = orbits_data.dropna()  #drop rows with missing values
cratio = 0.9  #percentage of data used for training
cutoff = round(cratio*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')
subset = ['Orbit Eccentricity','Perihelion Distance (AU)']   #what subset of features to use
train_subset = train_orbits_data[subset]
test_subset = test_orbits_data[subset]

#PREPROCESSING:

a_pipeline = Pipeline([('ala_scaler',StandardScaler()),('ala_svc',svm.SVC(kernel='poly'))])



# TRAINING AND TESTING MODEL
a_pipeline.fit(train_subset,train_labels)  #TRAIN MODEL


predictions = a_pipeline.predict(test_subset)   #PREDICT TESTING DATA
st.write("5-fold cross validation was computed on the training data. The accuracy ratio for each of the 5 folds is given below.")
accuracies = cross_val_score(a_pipeline,train_subset,train_labels, cv=5)
#accuracy = clf.score(test_subset,test_labels)       #OBTAIN ACCURACY
st.write(accuracies)
#st.write(np.unique(predictions))



#st.write(orbits_data['Object Classification'].value_counts())

st.write(confusion_matrix(test_labels,predictions))

#BUTTON DISPLAYS: the training and testing data using containers
#-----------------------------------------------------------------
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
            st.dataframe(train_subset)
    with traincol2:
        with traindataexpander2:
            st.dataframe(train_labels)
        
with test_container:
    with testcol1:
        with testdataexpander1:
            st.dataframe(test_subset)
    with testcol2:
        with testdataexpander2:
            st.dataframe(test_labels)

#------------------------------------------------------------

#plot decision boundaries


fig,ax = plt.subplots()

sns.scatterplot(data=orbits_data,x=subset[0],y=subset[1],hue = "Object Classification", palette = "muted",ax=ax)

DecisionBoundaryDisplay.from_estimator(
    a_pipeline,
    train_subset,
    plot_method="contour",
    colors="k",
    levels = [-1,0,1],
    alpha=0.5,
    linestyles = ["--","-", "--"],
    ax=ax
)
plt.show()
st.pyplot(fig)