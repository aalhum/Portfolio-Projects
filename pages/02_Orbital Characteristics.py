import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


st.write("Plotting of Known Asteroid Orbital Characteristics")

st.write("Information on the different near-earth object classifications can be found here: https://cneos.jpl.nasa.gov/about/neo_groups.html#:~:text=The%20vast%20majority%20of%20NEOs,%2Dmajor%20axes%20(%20a%20).")

st.write("These classifications are based on orbit location, perihelion distance, aphelion distance, and semi-major axis length.")
st.write("Potentially hazardous asteroids are classified as such based on Earth minimum orbit intersection distance as well as absolute magnitude.")

#read in the data using pandas read_csv()
orbits_data = pd.read_csv('orbits.csv')

raw_data_expand_2 = st.expander('Show Raw Data Table')


with raw_data_expand_2:
    st.dataframe(data=orbits_data)

#gotta figure out way to exclude the non-numeric variables from the drop-down lists for plotting

asteroid_class_stats = st.container()
#non-hazardous classification differences
not_hazard_orbit = pysqldf("SELECT * FROM orbits_data WHERE [Object Classification] NOT LIKE '%Hazard%';")

#hazardous classification differences
hazard_orbit = pysqldf("SELECT * FROM orbits_data WHERE [Object Classification] LIKE '%Hazard%';")

with asteroid_class_stats:
    st.write('Choose an orbital characteristic below to see how the boxplots differ by asteroid classification:')

    summary_class_param = st.selectbox(label='Choose a variable',options=orbits_data.columns[2::])
    
    fig_summ,ax_summ = plt.subplots()
    #I want the user to pick a variable to plot and then it generates scatterplot comparisons for the different asteroid classifications
    sns.boxplot(not_hazard_orbit,x='Object Classification',y=summary_class_param)  #plot boxplots for the nonhazardous asteroids with y being selected parameter
    st.pyplot(fig_summ)
    fig_summ2, ax_summ2 = plt.subplots()
    sns.boxplot(hazard_orbit,x='Object Classification',y=summary_class_param)
    st.pyplot(fig_summ2)

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

st.pyplot(sns.relplot(data=orbits_data, x=plot_var_1,y=plot_var_2, hue = "Object Classification", palette = "muted"))


plt.xlabel(plot_var_1)
plt.ylabel(plot_var_2)

st.write("Choose 2 asteroid parameters to plot")

