#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="combine.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        # Fill null values with column means
        self.data.fillna(self.data.mean(), inplace=True)


# In[ ]:




