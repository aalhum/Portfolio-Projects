import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xlsxwriter as xl
from sklearn import svm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

st.write("Export a scatterplot of your choice by selecting the plotting parameters.")

st.write("Select one of the models to run and export the plot to an excel spreadsheet")

col1, col2 = st.columns(2)

with col1:
    
    plot_var_1 = st.selectbox(label='Select x-axis',options=orbits_data.columns[1::])
    st.write(get_var_explanation(plot_var_1))   #show the written explanation for what this plotting parameter is

with col2:
    plot_var_2 = st.selectbox(label='Select y-axis',options=orbits_data.columns[1::])
    st.write(get_var_explanation(plot_var_2))

st.write("Choose 2 asteroid parameters to plot")
