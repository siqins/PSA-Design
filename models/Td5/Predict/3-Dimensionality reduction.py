import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle
model_path=r'transformer_3.24'
data_all = pd.read_csv('Test_filter_descriptors.csv', low_memory=True)

def trans(Data, model_path):
    with open(model_path, 'rb') as fw:
        model = pickle.load(fw)
        x_new= model.transform(Data)
        return x_new

# Locate descriptors for conversion
data = data_all.values[:, 1:]

data_New=trans(data,model_path)

# Save the X after dimensionality reduction
data_pd = pd.DataFrame(data_New)


data_pd.to_csv('Test_X.csv', index=False)
