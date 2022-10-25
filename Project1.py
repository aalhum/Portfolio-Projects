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


st.title("**Exploration of NASA Potential Asteroid Impact Data**")
st.write("Drake Khazal")
st.write("Ala Al-Humadi")

st.write("Plotting of NASA Sentry Asteroid Impact Data")

#read in the data using pandas read_csv()
impact_data = pd.read_csv('impacts.csv')
st.dataframe(data=impact_data)



#gotta figure out way to exclude the non-numeric variables from the drop-down lists for plotting

def get_explanation(plot_var):   #get a text description for the variable being plotted. note I wanted to use switch/case statements but don't think Python has them
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
        var_explain = 'How bright the object would appear to an observer if the asteroid was 1 au away from both the Earth and from the sun, at zero phase angle'
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
    st.write(get_explanation(plot_var_1))

with col2:
    plot_var_2 = st.selectbox(label='Select y-axis',options=impact_data.columns[1::])
    st.write(get_explanation(plot_var_2))



#create matplotlib figure and axis
fig,ax = plt.subplots()

#create scatterplot using variables from drop down menu
#the colormap is a built-in colormap, use it to generate a list of color values and assign those based on the absolute magnitude

st.pyplot(sns.relplot(data=impact_data, x=plot_var_1,y=plot_var_2, hue = "Asteroid Magnitude", palette="crest", sizes=(20,300), size = "Asteroid Diameter (km)"))


plt.xlabel(plot_var_1)
plt.ylabel(plot_var_2)

st.write("Choose 2 asteroid parameters to plot")

#ax.legend(s,'Asteroid Diameter (km)')


