import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle

def main():
    data_all = pd.read_csv('Test_filter_descriptors.csv', low_memory=True)
    data_pd_temp = pd.read_csv('Test_all_descriptors.csv', usecols=['RepeatUnitSMILES', 'td_value'])

    # Locate Descripotors
    X = data_all.values[:, 2:]

    pipe = Pipeline([('min_max', MinMaxScaler()),('PCA', PCA(n_components=0.95)), ('Transformer', PowerTransformer())])
    pipe.fit(X)
    X_New = pipe.transform(X)

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
