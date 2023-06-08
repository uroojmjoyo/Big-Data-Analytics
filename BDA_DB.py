#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd   
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from numpy import mean
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import DataLoader

import streamlit as st
import pickle

import os
from PIL import Image
import streamlit.components.v1 as components
#import io
import altair as alt


# In[19]:
data_loader = DataLoader()
data_loader.load_dataset()   
st.set_page_config(layout="wide")
image = Image.open(r"1601738239.jpg")

st.image(image)
st.title('World Development Analysis')
st.write('This is a web app to showcase demographic and financial visualization as part of BDA Project. Please adjust the value of each feature. After that, each country\'s analysis will appear.')

data_loader.columns = data_loader.columns.str.strip()

st.sidebar.title("Dashboard Information")
#st.sidebar.markdown("**Created by:** Urooj Mumtaz Joyo")
st.sidebar.markdown("**Data Source:** The World Bank")
st.sidebar.markdown("**Group Members:** Sarah Khalid, Urooj Mumtaz Joyo, Wajeeha Parker")
st.sidebar.markdown("**Project:** BDA - MLOPS - EcoTrend")

with st.sidebar:
    # Create a list of country names
    countries = data_loader['Country Name'].unique().tolist()
    indicators = data_loader['Indicator Name'].unique().tolist()
    # Country selection
    selected_country = st.selectbox('Select a Country', countries)
    selected_indicator = st.selectbox('Select an Indicator', indicators)
    selected_country2 = st.selectbox('Select Another Country for Comparison', countries)
    # Filter the data based on the selected country and indicator column
    filtered_df = data_loader[
        ((data_loader['Country Name'] == selected_country) |
        (data_loader['Country Name'] == selected_country2)) &
        (data_loader['Indicator Name'] == selected_indicator)]
    # Years column start from column 5 according to python 4
    years = filtered_df.columns[4:].tolist()
    # Year selection (multiple)
    selected_years = st.multiselect('Select years', years)
    # Filter the data based on the selected years
    filtered_df = filtered_df[['Country Name'] + selected_years]

# Melt the dataframe to reshape it
melted_df = pd.melt(filtered_df, id_vars='Country Name', var_name='Year', value_name='Value')

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
filtered_df_primary = melted_df[melted_df['Country Name'] == selected_country]
filtered_df_secondary = melted_df[melted_df['Country Name'] == selected_country2]
# Primary line chart
line_chart_primary = alt.Chart(filtered_df_primary).mark_line().encode(
    x='Year',
    y='Value',
    color=alt.Color('Country Name:N')  # Set the color of the primary line
)

# Filter the data for the secondary line chart
#filtered_df2 = melted_df[melted_df['Country Name'] == selected_country2]

# Secondary line chart
line_chart_secondary = alt.Chart(filtered_df_secondary).mark_line().encode(
    x='Year',
    y='Value',
    color=alt.Color('Country Name:N')  # Set the color of the secondary line
)

# Combine the line charts
line_chart_combined = alt.layer(line_chart_primary, line_chart_secondary).resolve_scale(
    y='independent'  # Use independent scales for the y-axis
)

# Configure the chart properties
line_chart_combined = line_chart_combined.properties(
    width=alt.Step(20),  # Set the width of the line
    title=selected_indicator
).configure_title(
    fontSize=14,
    anchor='start',
    color='#ff7f0e',
    offset=10
)

st.subheader('Bar Chart')
st.write('The graph contrasts the values of a chosen indicator for two countries over various time periods. The y-axis displays the values of the indicator, while the x-axis displays the years. The value of the indicator for each bar in the chart corresponds to a particular year and nation. When the cursor is over a bar, a tooltip with the year and related value appears. The two countries being compared are distinguished by the colour of the bars.')
st.altair_chart(bar_chart, use_container_width=True)

st.subheader('Line Graph')
st.write('A selected indicator\'s values for two countries over various years are compared in the resulting line chart. Two lines that each represent the indicator values for a different country make up the chart. The y-axis displays the values of the indicator, while the x-axis displays the years. Each line\'s colour indicates which of the two countries is being compared. The chosen indicator is used as the chart\'s title. The chart\'s distinct y-axis scales make it easier to compare the values of the two nations.')
st.altair_chart(line_chart_combined, use_container_width=True)


# In[20]:


st.title('Prediction')

st.sidebar.markdown('**Prediction**')
st.write('Please select the country, indicator and year for which you want to predict.')

with st.sidebar:
    # Create a list of country codes and indicator codes
    countries = data_loader['Country Code'].unique().tolist()
    indicators = data_loader['Indicator Code'].unique().tolist()

    # Country code to name mapping
    country_names = data_loader[['Country Code', 'Country Name']].drop_duplicates().set_index('Country Code')['Country Name'].to_dict()

    # Indicator code to name mapping
    indicator_names = data_loader[['Indicator Code', 'Indicator Name']].drop_duplicates().set_index('Indicator Code')['Indicator Name'].to_dict()

    years = ['2022', '2023', '2024', '2025', '2026', '2027']
    # Country selection
    selected_country = st.selectbox('Select a Country for Prediction', countries, format_func=lambda x: country_names[x])
    selected_indicator = st.selectbox('Select an Indicator for Prediction', indicators, format_func=lambda x: indicator_names[x])
    selected_year = st.selectbox('Select a Year for Prediction', years)

data_loader = data_loader[data_loader["Indicator Code"].isin([selected_indicator])]  # Changed from 'Indicator Name'
id_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']  # Changed from 'Indicator Name'
data_loader = pd.melt(df, id_vars=id_columns, var_name='Year', value_name='Value')
data_loader = data_loader.sort_values(['Country Name', 'Year'])
data_loader = data_loader.reset_index(drop=True)
data_loader = data_loader[data_loader["Country Code"].isin([selected_country])]

X = data_loader.drop(columns=['Country Name', 'Indicator Name', 'Value'])  # Changed from 'Indicator Name'
X['Indicator Code'] = LabelEncoder().fit_transform(X['Indicator Code'])  # Encode the Indicator Code
X['Country Code'] = LabelEncoder().fit_transform(X['Country Code'])
X = X.astype(int)
y = data_loader[['Value']]

Xt = X.copy()
Xt['Year'] = int(selected_year)

gb = GradientBoostingRegressor(max_depth=5, max_features=4, n_estimators=2300, learning_rate=0.056, random_state=0)
xgb = XGBRegressor(n_estimators=1300, learning_rate=0.04, n_jobs=-1)
lgbm = lgb.LGBMRegressor(learning_rate=0.09, n_estimators=5800, max_depth=4)
vot = VotingRegressor([('gb', gb), ('xgb', xgb), ('lgb', lgbm)])

reg = vot.fit(X, y)

features = {
    'Country Code': LabelEncoder().fit_transform([selected_country])[0],
    'Indicator Code': LabelEncoder().fit_transform([selected_indicator])[0],
    'Year': int(selected_year)
}

st.write("Selected Features:")
st.write(pd.DataFrame([features]))

col1, col2 = st.columns((1, 2))

with col1:
    prButton = st.button('Predict')

with col2:
    if prButton:
        features_df = pd.DataFrame([features])
        prediction = reg.predict(features_df)
        st.write('Based on the selected features, the predicted value is: {}'.format(int(prediction)))

# Line Chart
st.write("Line Chart:")
selected_country_data = data_loader[data_loader['Country Code'] == selected_country]
selected_country_data = selected_country_data[selected_country_data['Indicator Code'] == selected_indicator]  # Changed from 'Indicator Name'
selected_country_data = selected_country_data[selected_country_data['Year'].astype(int) <= int(selected_year)]
line_chart = pd.DataFrame(selected_country_data[['Year', 'Value']])
line_chart = line_chart.set_index('Year')
st.line_chart(line_chart)


# In[ ]:




