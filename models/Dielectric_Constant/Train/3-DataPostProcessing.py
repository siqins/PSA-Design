import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import pickle

def main():
    data_all = pd.read_csv('Test_filter_descriptors.csv', low_memory=True)
    data_pd_temp = pd.read_csv('Test_all_descriptors.csv', usecols=['RepeatUnitSMILES', 'PropertyValue', 'TestFreq', 'TestTemp'])

    # Locate Descripotors
    X = data_all.values[:, 4:]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    pipe = Pipeline([('PCA', PCA(n_components=0.9)), ('Transformer', PowerTransformer())])
    pipe.fit(X_minmax)
    X_New = pipe.transform(X_minmax)

    # Locate Descripotors
    data_pd = pd.DataFrame(X_New)

    # Merge SMILES-Property-Freq-Temp-X
    data_pd_out = pd.concat([data_pd_temp, data_pd], axis=1)

    # Save file
    data_pd_out.to_csv('Test_SMILES_Y_X.csv', index=False)

    # Save the training converter
    with open('transformer', 'wb') as fw:
        pickle.dump(pipe, fw)


if __name__ == '__main__':
    main()
