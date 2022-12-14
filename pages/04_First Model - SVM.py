import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
from sklearn.feature_selection import RFECV, SelectKBest, chi2

#citation: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

#CODE FOR FIRST MODEL - SVM

st.title('Support Vector Machine')

st.write("A Support Vector Machine (SVM) is an algorithm that is best for linearly separable data. It can be used for both classification and regression. It works by selecting \"support vectors\", which are a subset of the data points chosen to represent their classes. The algorithm separates the data points into classes by dividing them using a decision boundary, with the largest possible margin on either side. The margin boundaries are on the support vectors. ")


#SET UP TRAINING AND TESTING DATA
orbits_data = pd.read_csv('orbits.csv')  #read in orbits data
#st.write(labels.unique())    #check how many unique values are in asteroid classification
orbits_data = orbits_data.drop(['Object Name','Orbital Reference'],axis=1) #drop columns not used as features
orbits_data = orbits_data.dropna()  #drop rows with missing values
cratio = 0.75  #percentage of data used for training
cratio_example = 0.05 #use a trivial number of the samples just for visualization purposes
cutoff = round(cratio*orbits_data.shape[0])  #cutoff index that separates training and testing data, 90% training, 10% test
test_orbits_data = orbits_data.iloc[cutoff::,:] 
train_orbits_data = orbits_data.iloc[0:cutoff,:]
test_labels = test_orbits_data.pop('Object Classification')  #get the asteroid classifications/target labels
train_labels = train_orbits_data.pop('Object Classification')

allowed_subset = ['Orbit Axis (AU)','Orbit Eccentricity','Orbit Inclination (deg)','Perihelion Argument (deg)','Node Longitude (deg)','Mean Anomoly (deg)','Orbital Period (yr)']  #list of possible features

allowed_train = train_orbits_data[allowed_subset]   ##FINAL TRAIN/TEST FEATURES after dropping off more features
allowed_test = test_orbits_data[allowed_subset]

optimal_features = ['Perihelion Argument (deg)','Mean Anomoly (deg)','Orbital Period (yr)']

st.write('First, we will demonstrate how the SVM generates a decision boundary below by plotting only 2 of the features at a time so they can be displayed on a graph.')
st.write('Select 2 features of the data to plot. Then choose the number of output classes - this is how many categories/groups that the data points belong to - do you want to see all of the asteroid classifications, or combine Hazardous and Non-Hazardous asteroids together, or simply see whether asteroids are Hazardous or Non-Hazardous?')
st.write('Finally, choose a kernel, which is the shape of the decision boundary -- should the decision boundary be linear or can it be curved?')

st.write('The two features you select will be plotted below, with the points being assigned different colors based on their asteroid classification (see the legend)')
st.write('You will see the decision boundary as a segmented black line that attempts to separate the different classes from each other')

    
colum1,colum2 = st.columns(2)
#allow user to select two features to plot in example plot
with colum1:
    svmparameter1 = st.selectbox(label='Choose 1st Feature',options=allowed_train.columns)
with colum2:
    svmparameter2 = st.selectbox(label='Choose 2cd Feature',options=allowed_train.columns,index=1)

subset = [svmparameter1,svmparameter2]   #what subset of features to use

same_parameter = False

if svmparameter1 == svmparameter2:
    st.write("You have selected the same feature twice! Please select 2 different features")
    same_parameter = True

st.write("Select Number of Output Classes:")

train_subset = allowed_train[subset]   #example train/test sets with only 2 features
test_subset = allowed_test[subset]

num_classes = st.selectbox(label='Select the Number of Output Classes',options=['All Classes','4 Classes - No Hazard','Binary - Hazard/Not Hazard'])

kernel_type = st.selectbox(label='Select Kernel Type',options=['linear','poly','rbf','sigmoid'])

num_degree = st.selectbox(label='Select Degree (for Polynomial kernel)',options=[1,2,3,4])

example_cutoff = round(cratio_example*train_subset.shape[0])   #use the example cutoff instead of the actual cutoff
example_train_subset = train_subset.iloc[0:example_cutoff,:]
all_labels = train_labels.iloc[0:example_cutoff]

if num_classes == 'All Classes':
    final_labels = all_labels
elif num_classes == '4 Classes - No Hazard':
    quad_labels = all_labels.copy()
    final_labels = quad_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})
elif num_classes == 'Binary - Hazard/Not Hazard':
    binary_labels = all_labels.copy()
    final_labels = binary_labels.replace({'Amor Asteroid (Hazard)':'Hazard','Apollo Asteroid (Hazard)':'Hazard','Apohele Asteroid (Hazard)':'Hazard','Aten Asteroid (Hazard)':'Hazard','Amor Asteroid':'Not Hazard','Apollo Asteroid':'Not Hazard','Aten Asteroid':'Not Hazard','Apohele Asteroid':'Not Hazard'})

#st.write("Possible Output Labels:")
#st.write(np.unique(np.asarray(final_labels)))


data_and_labels = example_train_subset.copy()
data_and_labels['Object Classification'] = final_labels  #add the class labels to the data so you can plot them
#Create Pipeline that includes preprocessing + SVM model

a_pipeline = Pipeline([('ala_scaler',StandardScaler()),('ala_svc',svm.SVC(kernel=kernel_type,degree = num_degree))])


# TRAINING THE MODEL for the example
if same_parameter == False:
    a_pipeline.fit(example_train_subset,final_labels)  #TRAIN MODEL



fig,ax = plt.subplots()

if same_parameter == False: sns.scatterplot(data=data_and_labels,x=subset[0],y=subset[1],hue = "Object Classification", palette = "muted",ax=ax)

if same_parameter == False:
    DecisionBoundaryDisplay.from_estimator(
        a_pipeline,
        example_train_subset,
        plot_method="contour",
        colors="k",
        levels = [-1,0,1],
        alpha=0.5,
        linestyles = ["--","-", "--"],
        ax=ax
)

st.pyplot(fig)

#to do: figure out what classes to display/predict,
#add prediction using the full subset of data and then using just the optimal features
#do the same for the other features
#perhaps try other feature selection methods to compare?


#RUNNING THE ALGORITHM FOR REAL, WITH CHOSEN FEATURES - JUST REPLACE THE FEATURES BELOW
#now do hyper parameter tuning using gridsearchCV
#allowed_train
#allowed_test
train_labels_4 = train_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})
test_labels_4 = test_labels.replace({'Amor Asteroid (Hazard)':'Amor Asteroid','Apollo Asteroid (Hazard)':'Apollo Asteroid','Apohele Asteroid (Hazard)':'Apohele Asteroid','Aten Asteroid (Hazard)':'Aten Asteroid'})

st.write('When we use all of the features available to predict 4 asteroid classes, the resulting accuracy is:')
b_pipeline = Pipeline([('ala_scaler2',StandardScaler()),('ala_svc2',svm.SVC(kernel='rbf'))])
#fit the model
b_pipeline.fit(allowed_train,train_labels_4)

#predict on test set
accuracy_full = b_pipeline.score(allowed_test,test_labels_4)

st.write("The accuracy of the SVM with rbf kernel on the test data is:")
st.write(accuracy_full)
#97.3% accuracy with rbf kernel

#at the very least record these values using all of the features for each model, and put it into a table at the end - it's useful