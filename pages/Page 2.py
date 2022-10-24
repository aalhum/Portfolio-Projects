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

#create two column containers and put a selectbox in each one
col1, col2 = st.columns(2)

with col1:
    
    plot_var_1 = st.selectbox(label='Select x-axis',options=orbits_data.columns[2::])

with col2:
    plot_var_2 = st.selectbox(label='Select y-axis',options=orbits_data.columns[2::])


#create matplotlib figure and axis
fig,ax = plt.subplots()

#create scatterplot using variables from drop down menu
#the colormap is a built-in colormap, use it to generate a list of color values and assign those based on the absolute magnitude

st.pyplot(sns.relplot(data=orbits_data, x=plot_var_1,y=plot_var_2, hue = "Asteroid Magnitude"))


plt.xlabel(plot_var_1)
plt.ylabel(plot_var_2)

st.write("Choose 2 asteroid parameters to plot")