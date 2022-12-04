import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xlsxwriter as xl
from sklearn import svm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

st.write("If you have a CSV ")

workbook = xl.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1','Hello world')
workbook.close()
st.write(workbook)
asteroid_excel = st.download_button("Download Excel Visualization of Data",workbook,file_name='Asteroid Data')
