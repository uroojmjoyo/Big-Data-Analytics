#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="combine.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        self.data =  self.data.drop(columns='2022')
        # Fill null values with column mean
        columns_with_missing_values = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", 
                               "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
                               "2020", "2021"]  # Replace with your column names
        # Replace '-' values with NaN
        self.data[columns_with_missing_values] = self.data[columns_with_missing_values].replace('-', np.nan)
        # Convert the columns with missing values to numeric type
        self.data[columns_with_missing_values] = self.data[columns_with_missing_values].astype(float)
        # Fill missing values with the respective mean values
        for column in columns_with_missing_values:
            mean_value = self.data[column].mean()
            self.data[column].fillna(mean_value, inplace=True)

# In[ ]:




