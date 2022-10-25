import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



st.write("Plotting of Known Asteroid Orbital Characteristics")

#read in the data using pandas read_csv()
orbits_data = pd.read_csv('orbits.csv')
st.dataframe(data=orbits_data)


#gotta figure out way to exclude the non-numeric variables from the drop-down lists for plotting

def get_explanation(plot_var):   #get a text description for the variable being plotted. note I wanted to use switch/case statements but don't think Python has them
    if plot_var == 'Epoch (TDB)':
        var_explain = 'Epoch is a time point used as reference for the other orbital characteristics.'
    elif plot_var == 'Orbit Axis (AU)':
        var_explain = ''
    elif plot_var == 'Orbit Eccentricity':
        var_explain = 'Length of the semi-major axis for the elliptical orbit'
    elif plot_var == 'Orbit Inclination (deg)':
        var_explain = 'The tilt or orientation of the orbit axis relative to Earth\'s orbit, in degrees'
    elif plot_var == 'Perihelion Argument (deg)':
        var_explain = 'The angle of the orbiting body as it crosses the perihelion, or closest point to the sun on its orbit'
    elif plot_var == 'Node Longitude (deg)':
        var_explain = 'The angle of the orbiting body as it crosses Earth\s orbit, or ecliptic orbit'
    elif plot_var == 'Mean Anomaly (deg)':
        var_explain = 'The fraction of time spent in the orbit since the orbiting body passed periapsis'
    elif plot_var == 'Perihelion Distance (AU)':
        var_explain = 'The closest distance on the asteroid\'s orbit path to the sun'
    elif plot_var == 'Aphelion Distance (AU)':
        var_explain = 'The farthest distance on the asteroid\'s orbit from the sun'
    elif plot_var == 'Orbital Period (yr)':
        var_explain = 'The time that it takes to complete a full orbital revolution.'
    elif plot_var == 'Minimum Orbit Intersection Distance (AU)':
        var_explain = 'The distance between the two closest points on the orbits of two bodies'
    elif plot_var == 'Orbital Reference':
        var_explain = 'ID'
    elif plot_var == 'Asteroid Magnitude':
        var_explain = 'How bright the object would appear to an observer if the asteroid was 1 au away from both the Earth and from the sun, at zero phase angle'
    else:
        var_explain = 'No Parameter Selected'
        
    return var_explain

#create two column containers and put a selectbox in each one
col1, col2 = st.columns(2)

with col1:
    plot_var_1 = st.selectbox(label='Select x-axis',options=orbits_data.columns[2::])
    st.write(get_explanation(plot_var_1))

with col2:
    plot_var_2 = st.selectbox(label='Select y-axis',options=orbits_data.columns[2::])
    st.write(get_explanation(plot_var_2))

#create matplotlib figure and axis
fig,ax = plt.subplots()

#create scatterplot using variables from drop down menu
#the colormap is a built-in colormap, use it to generate a list of color values and assign those based on the absolute magnitude

st.pyplot(sns.relplot(data=orbits_data, x=plot_var_1,y=plot_var_2, hue = "Asteroid Magnitude"))


plt.xlabel(plot_var_1)
plt.ylabel(plot_var_2)

st.write("Choose 2 asteroid parameters to plot")