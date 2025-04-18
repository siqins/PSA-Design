#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


# 1.Remove the column that reported the calculation error
# 2.Remove all zero columns

if __name__ == '__main__':

    # Read data
    data_pd = pd.read_csv('Test_all_descriptors.csv', low_memory=False)
    data_pd_temp = pd.read_csv('Test_all_descriptors.csv', usecols=['RepeatUnitSMILES', 'td_value'])
    # print(data_pd)

    # Get a list of column names
    column_list = [column for column in data_pd]

    # Convert to numpy array
    data_np = data_pd.values
    # print(data_np)
    m, n = data_np.shape
    # print(m, n)

    # Traversal / 4: each column (0,1,2,3 for SMILES, Property, Freq, Temp).
    # If there is a value in the column that is not 0, delete it. If the value is 0, delete it
    filter_descriptors = {}
    for i in range(4, n):
        tag = 0
        sum = 0

        # Gets the column name (descriptor name) and all values for that column
        des_Name = column_list[i]
        des_value_column = data_np[:, i].tolist()

        # Whether the two values are True or False is ignored
        if des_Name == 'GhoseFilter' or des_Name == 'Lipinski':
            continue

        # All values of the descriptor are traversed for processing
        for value in des_value_column:
            if isinstance(value, str):
                tag = 1
                break
            if np.isnan(value):
                tag = 1
                break
            sum += abs(value)
        if tag == 1:
            continue
        if sum == 0:
            continue

        # Save descriptor columns that meet the condition
        filter_descriptors[des_Name] = des_value_column

    filter_descriptors_pd = pd.DataFrame(filter_descriptors)
    print(filter_descriptors_pd)

    # Merge SMILES-Property-Freq-Temp-FilterDescriptors
    data_pd_out = pd.concat([data_pd_temp, filter_descriptors_pd], axis=1)

    # Save file
    data_pd_out.to_csv('Test_filter_descriptors.csv', index=False)
