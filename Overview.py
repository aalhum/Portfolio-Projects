#Data Science Project 1
#Ala Al-Humadi
#Drake Khazal

#import statements
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import linear_model
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

st.title("**Exploration of NASA Potential Asteroid Impact Data**")

st.write("Contributors:")

st.write("Ala Al-Humadi")
st.write("With Assistance from: Drake Khazal")



st.write("The data used for this project was taken from: https://www.kaggle.com/datasets/nasa/asteroid-impacts")
st.write("The data is in 2 parts:")
st.write("The first dataset is displayed on this page below, and it includes variables such as Asteroid Magnitude (a measure of brightness), Asteroid Velocity, Palermo/Torino Scale values (click on these parameters in the drop-down menu to learn more about them)")
st.write("The second dataset is displayed on the \"Orbital Characteristics\" page. It lists the names of many known asteroids and their orbital characteristics, such as orbit eccentricity, node longitude, perihelion/aphelion distance, etc.... Select each variable in the drop-down menu on that page to learn more")
st.write("The data is from NASA's Sentry system, a long-term system that monitors the potential orbits of asteroids to see whether they will crash into Earth (a very unlikely occurrence).")
st.write("On these first two pages, we plot the data using a scatterplot to observe the general trends ")

st.write("We decided to try the SVM model because the SVM algorithm is good for linearly separable data so we were curious and wanted to compare other methods that can better represent nonlinear decision boundaries (as well as different SVM nonlinear kernels).")
st.write("We decided to try KNN as it is a very simple algorithm that is a good point of comparison to the other more involved algorithms.")
st.write("Finally, we decided to try using Random Forest because it is one example of an ensemble method utilizing an underlying very simple model - a decision tree, so we expected it to perform better than SVM and KNN.")
#read in the data using pandas read_csv()
impact_data = pd.read_csv('impacts.csv')

raw_data_expand_1 = st.expander('Show Raw Data Table')
summary_expand = st.expander('Show Summary Statistics for Dataset')

with raw_data_expand_1:
    st.dataframe(data=impact_data)
    
with summary_expand:
    st.write(impact_data.describe())



#gotta figure out way to exclude the non-numeric variables from the drop-down lists for plotting

def get_var_explanation(plot_var):   #get a text description for the variable being plotted. note I wanted to use switch/case statements but don't think Python has them
    if plot_var == 'Period Start':
        var_explain = 'The start time for the asteroid\'s orbit'
    elif plot_var == 'Period End':
        var_explain = 'The end time for the asteroid\'s orbit'
    elif plot_var == 'Possible Impacts':
        var_explain = 'Number of calculated potential times that the asteroid could impact Earth'
    elif plot_var == 'Cumulative Impact Probability':
        var_explain = 'The cumulative probability from all of the possible impacts that the asteroid could have'
    elif plot_var == 'Asteroid Velocity':
        var_explain = 'The velocity of the asteroid relative to Earth\'s velocity, assuming Earth doesn\'t have mass.'
    elif plot_var == 'Asteroid Magnitude':
        var_explain = 'How bright the object would appear to an observer if the asteroid was 1 au away from both the Earth and from the sun, at zero phase angle. NOTE: The lower the magnitude, the brighter the object'
    elif plot_var == 'Asteroid Diameter (km)':
        var_explain = 'The diameter of the asteroid in km'
    elif plot_var == 'Cumulative Palermo Scale':
        var_explain = 'The Cumulative Palermo Technical Impact Hazard Scale is a scale for describing how likely a potential asteroid impact is. This is done by calculating \'background risk\', or what the average risk of random asteroid impacts in all the years leading up to this particular asteroid\'s impact. The scale is logarithmic. The more negative the Palermo Scale value is, the less likely it is to potentially impact Earth, with 0 representing the background level of risk. Positive values are very rare.'
    elif plot_var == 'Maximum Palermo Scale':
        var_explain = 'See \'Cumulative\' Palermo Scale for full explanation of Palermo Scale. This is the max value on the Hazard Scale.'
    elif plot_var == 'Maximum Torino Scale':
        var_explain = 'The Torino Scale is meant for general public perception of asteroid impact likelihood. The Torino Scale ranges from 0 to 10, with higher values meaning more likely the asteroid will impact'
    else:
        var_explain = 'No Parameter Selected'
        
    return var_explain

#create two column containers and put a selectbox in each one
col1, col2 = st.columns(2)

with col1:
    
    plot_var_1 = st.selectbox(label='Select x-axis',options=impact_data.columns[1::])
    st.write(get_var_explanation(plot_var_1))   #show the written explanation for what this plotting parameter is

with col2:
    plot_var_2 = st.selectbox(label='Select y-axis',options=impact_data.columns[1::])
    st.write(get_var_explanation(plot_var_2))

st.write("Choose 2 asteroid parameters to plot")

#create matplotlib figure and axis
fig,ax = plt.subplots()

#create scatterplot using variables from drop down menu
#the colormap is a built-in colormap, use it to generate a list of color values and assign those based on the absolute magnitude

st.pyplot(sns.relplot(data=impact_data, x=plot_var_1,y=plot_var_2, hue = "Asteroid Magnitude", palette="crest", sizes=(20,300), size = "Asteroid Diameter (km)"))


plt.xlabel(plot_var_1)
plt.ylabel(plot_var_2)

def get_plot_explanation(pvar1,pvar2):
    if pvar1 == pvar2:
        plot_explanation = "You have selected the same variable for both x- and y-axis"
        
    elif pvar1 in ['Asteroid Velocity','Cumulative Impact Probability'] and pvar2 in ['Asteroid Velocity','Cumulative Impact Probability']:   #for first parameter of asteroid velocity, go through the list of second parameters
            plot_explanation = "There are a wide spread of asteroid velocities but there does not seem to be an appreciable pattern or change from the cumulative impact probability being near 0, except for one outlier with a velocity near 6 and a probability above 0.06"
    elif pvar1 in ['Asteroid Velocity','Possible Impacts'] and pvar2 in ['Asteroid Velocity','Possible Impacts']:
            plot_explanation = "The asteroids with lower velocities in general seem to have a higher spread of possible predicted impacts, and the ones with the highest possible impacts appear to have higher magnitudes (i.e darker) and lower asteroid diameter"
    elif pvar1 in ['Asteroid Velocity','Asteroid Magnitude'] and pvar2 in ['Asteroid Velocity','Asteroid Magnitude']:
            plot_explanation = "There is a large spread of asteroids across both magnitude and velocity"
    elif pvar1 in ['Asteroid Velocity','Asteroid Diameter (km)'] and pvar2 in ['Asteroid Velocity','Asteroid Diameter (km)']:
            plot_explanation = 'Many of the asteroids with higher diameters have velocities in the middle of the range, around 10-25. Most of the asteroids have small diameters and have a wide spread of velocities'
    elif pvar1 in ['Asteroid Velocity','Cumulative Palermo Scale'] and pvar2 in ['Asteroid Velocity','Cumulative Palermo Scale']:
            plot_explanation = 'There is no clear pattern between asteroid velocity and its cumulative rating on the Palermo Scale; there is a wide spread. The only possible feature might be that the larger asteroid diameters appear to be higher up on the cumulative Palermo Scale.'
    elif pvar1 in ['Asteroid Velocity','Maximum Palermo Scale'] and pvar2 in ['Asteroid Velocity','Maximum Palermo Scale']:
            plot_explanation = 'There is no clear pattern between asteroid velocity and its max rating on the Palermo Scale. Some of the larger diameter asteroids seem to have higher max Palermo Scale values.'
    elif pvar1 in ['Asteroid Velocity','Maximum Torino Scale'] and pvar2 in ['Asteroid Velocity','Maximum Torino Scale']:
            plot_explanation = 'All of the Maximum Torino Scale values (except for 3 unknown values marked with \'*\') are 0, which is understandable when considering that none of these asteroids are likely to impact Earth at all.'
    
    elif pvar1 in ['Asteroid Magnitude','Possible Impacts'] and pvar2 in ['Asteroid Magnitude','Possible Impacts']:
        plot_explanation = 'As the Asteroid Magnitude increases, the number of possible impacts increases drastically, and it seems that many of the higher magnitude asteroids have smaller diameters'
    elif pvar1 in ['Asteroid Magnitude','Cumulative Impact Probability'] and pvar2 in ['Asteroid Magnitude','Cumulative Impact Probability']:
        plot_explanation = 'Virtually all of the asteroids have very low impact probabilities, with the exception of an outlier with a magnitude of about 28 and a probability of 0.06'
    elif pvar1 in ['Asteroid Magnitude','Asteroid Diameter (km)'] and pvar2 in ['Asteroid Magnitude','Asteroid Diameter (km)']:
        plot_explanation = 'There is a very clear exponentially decaying pattern between Asteroid Magnitude and Asteroid Diameter. This implies that as the asteroid\'s diameter increases, the asteroid appears brighter in magnitude, which makes sense.'
    elif pvar1 in ['Asteroid Magnitude','Cumulative Palermo Scale'] and pvar2 in ['Asteroid Magnitude','Cumulative Palermo Scale']:
        plot_explanation = 'There is a slight downward trend in the plot; the Cumulative Palermo Scale decreases as the asteroid magnitude increases.'
    elif pvar1 in ['Asteroid Magnitude','Maximum Palermo Scale'] and pvar2 in ['Asteroid Magnitude','Maximum Palermo Scale']:
        plot_explanation = 'As the asteroid magnitude increases, the Max Palermo Scale decreases, with many of the smaller-diameter/larger magnitude asteroids having lower max Palermo Scale values.'
    elif pvar1 in ['Asteroid Magnitude','Maximum Torino Scale'] and pvar2 in ['Asteroid Magnitude','Maximum Torino Scale']:
        plot_explanation = 'All of the Torino Scale values are 0 to communicate 0 likelihood to the public of an asteroid impact, except for 3 unknown values marked with \'*\''
    
    elif pvar1 in ['Asteroid Diameter (km)','Possible Impacts'] and pvar2 in ['Asteroid Diameter (km)','Possible Impacts']:
        plot_explanation = 'It seems that the smaller diameter asteroids have a much larger range of possible impact numbers, and the larger diameter asteroids generally have very low possible impacts.'
    elif pvar1 in ['Asteroid Diameter (km)','Cumulative Impact Probability'] and pvar2 in ['Asteroid Diameter (km)','Cumulative Impact Probability']:
        plot_explanation = 'All of the asteroids except for a couple outliers have very low cumulative impact probability. There is one outlier with a probability above 0.06, with a diameter of near 0'
    elif pvar1 in ['Asteroid Diameter (km)','Asteroid Magnitude'] and pvar2 in ['Asteroid Diameter (km)','Asteroid Magnitude']:
        plot_explanation = 'As the asteroid diameter increases, the asteroid magnitude or apparent brightness decreases in a quadratic manner, which corresponds to a brighter object. This makes sense as one would expect the asteroid magnitude/brightness to increase/be brighter as the size of the asteroid increases.'
    elif pvar1 in ['Asteroid Diameter (km)','Cumulative Palermo Scale'] and pvar2 in ['Asteroid Diameter (km)','Cumulative Palermo Scale']:
        plot_explanation = 'As the asteroid diameter increases, it appears that the higher diameter asteroids have higher cumulative Palermo Scale values.'
    elif pvar1 in ['Asteroid Diameter (km)','Maximum Palermo Scale'] and pvar2 in ['Asteroid Diameter (km)','Maximum Palermo Scale']:
        plot_explanation = 'The vast majority of the smaller asteroids have a wide range of Max Palermo Scale values. The larger diameter asteroids appear to have a higher max Palermo Scale.'
    elif pvar1 in ['Asteroid Diameter (km)','Maximum Torino Scale'] and pvar2 in ['Asteroid Diameter (km)','Maximum Torino Scale']:
        plot_explanation = 'All of the Torino Scale values are 0 to communicate 0 likelihood to the public of an asteroid impact, except for 3 unknown values marked with \'*\''
    
    elif pvar1 in ['Cumulative Palermo Scale','Possible Impacts'] and pvar2 in ['Cumulative Palermo Scale','Possible Impacts']:
        plot_explanation = 'The asteroids with middle-of-the-pack cumulative Palermo Scale seem to have higher possible numbers of calculated impacts, which seems to be an odd shape. The highest number of possible impacts appears to be near 1200.'
    elif pvar1 in ['Cumulative Palermo Scale','Cumulative Impact Probability'] and pvar2 in ['Cumulative Palermo Scale','Cumulative Impact Probability']:
        plot_explanation = 'The vast majority of asteroids have cumulative impact probabilities very close to 0.'
    elif pvar1 in ['Cumulative Palermo Scale','Maximum Palermo Scale'] and pvar2 in ['Cumulative Palermo Scale','Maximum Palermo Scale']:
        plot_explanation = 'There is a strong positive linear correlation between cumulative Palermo Scale score and Maximum Palermo Scale - this makes a lot of sense, as the maximum score probably contributes to the cumulative score.'
    elif pvar1 in ['Cumulative Palermo Scale','Maximum Torino Scale'] and pvar2 in ['Cumulative Palermo Scale','Maximum Torino Scale']:
        plot_explanation = 'All of the Torino Scale values are 0 to communicate 0 likelihood to the public of an asteroid impact, except for 3 unknown values marked wiith \'*\''
    
    elif pvar1 in ['Maximum Palermo Scale','Cumulative Palermo Scale'] and pvar2 in ['Maximum Palermo Scale','Cumulative Palermo Scale']:
        plot_explanation = 'Maximum Palermo Scale and Cumulative Palermo Scale are strongly positively linearly correlated, which makes sense as the max Palermo Scale value likely contributes to the cumulative value.'
    elif pvar1 in ['Maximum Palermo Scale','Maximum Torino Scale'] and pvar2 in ['Maximum Palermo Scale','Maximum Torino Scale']:
        plot_explanation = 'All of the Torino Scale values are 0 to communicate 0 likelihood to the public of an asteroid impact, except for 3 unknown values marked with \'*\''
    
    else:
        plot_explanation = "No Explanation"
    return plot_explanation
    
st.write("Plot Explanation")
st.write(get_plot_explanation(plot_var_1,plot_var_2))

#ax.legend(s,'Asteroid Diameter (km)')




