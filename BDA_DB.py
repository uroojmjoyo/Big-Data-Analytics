#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd   
import matplotlib.pyplot as plt
import time

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score


from numpy import mean
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go


from scipy.stats import randint as sp_randint, uniform as sp_uniform

import time

from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from interpret.blackbox import LimeTabular
from interpret import show
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import shap
# Run initjs() function
shap.initjs()

import dice_ml
import joblib

import streamlit as st
import pickle
import datetime
import os
from PIL import Image
import streamlit.components.v1 as components
import io
import altair as alt
from Utils1 import DataLoader


# In[ ]:


data_loader = DataLoader()
data_loader.load_dataset()

st.set_page_config(layout="wide")
image = Image.open(r"1601738239.jpg")

st.image(image)

#filename = 'WTI_OilPriceP.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

st.title('World Trade Analysis')
st.write('This is a web app to showcase trade visualization as part of BDA Project. Please adjust the value of each feature. After that, each country\'s analysis will appear.')

df.columns = df.columns.str.strip()

st.sidebar.title("Dashboard Information")
#st.sidebar.markdown("**Created by:** Urooj Mumtaz Joyo")
st.sidebar.markdown("**Group Members:** Sarah Khalid, Urooj Mumtaz Joyo, Wajeeha Parker")
st.sidebar.markdown("**Project:** BDA - MLOPS")

with st.sidebar:
    # Create a list of country names
    countries = df['Country Name'].unique().tolist()
    indicators = df['Indicator Name'].unique().tolist()
    # Country selection
    selected_country = st.selectbox('Select a Country', countries)
    selected_indicator = st.selectbox('Select an Indicator', indicators)
    #selected_country2 = st.selectbox('Select a Another Country for Comparison', countries)
    # Filter the data based on the selected country and indicator column
    filtered_df = df[
        (df['Country Name'] == selected_country) &
        #(df['Country Name'] == selected_country2)
        (df['Indicator Name'] == selected_indicator)]
    # Years column start from column 5 according to python 4
    years = filtered_df.columns[4:].tolist()
    # Year selection (multiple)
    selected_years = st.multiselect('Select years', years)
    # Filter the data based on the selected years
    filtered_df = filtered_df[['Country Name'] + selected_years]

# Melt the dataframe to reshape it
melted_df = pd.melt(filtered_df, id_vars='Country Name', var_name='Year', value_name='Value')

# Filter the data based on the selected country and indicator column
#line_df = df[
    #(df['Country Name'] == selected_country) &
    #(df['Indicator Name'] == selected_indicator)]

# Melt the dataframe to reshape it
#melted_line_df = pd.melt(line_df, id_vars='Country Name', var_name='Year', value_name='Value')

# Bar Chart
bar_chart = alt.Chart(melted_df).mark_bar().encode(
    x='Year',
    y='Value',
    tooltip=['Year', 'Value'],
    color=alt.Color('Country Name:N')
).properties(
    width=alt.Step(20)  # Set the width of the bars
).configure_title(
    fontSize=14,
    anchor='start',
    color='#1f77b4',
    offset=10
).properties(
    title=selected_indicator,
    width=500,
    height=300
)

# Line Graph
line_chart = alt.Chart(melted_df).mark_line().encode(
    x='Year',
    y='Value',
    color=alt.value('#ff7f0e')  # Set the color of the line
).properties(
    width=alt.Step(20)  # Set the width of the line
).configure_title(
    fontSize=14,
    anchor='start',
    color='#ff7f0e',
    offset=10
).properties(
    title=selected_indicator)


st.subheader('Bar Chart')
st.altair_chart(bar_chart, use_container_width=True)

st.subheader('Line Graph')
st.altair_chart(line_chart, use_container_width=True)

