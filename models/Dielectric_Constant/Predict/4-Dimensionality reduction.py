import pandas as pd
import pickle

# This model is the Pipeline processed by descriptors during model training, that is,
# the dimensionality reduction process. After normalization, the dimensionality is reduced to 10 dimensions by PCA
model_path=r'transformer'
data_all = pd.read_csv('Test_filter_descriptors.csv', low_memory=True)
data_pd_temp = pd.read_csv('Test_filter_descriptors.csv', usecols=['TestFreq', 'TestTemp'])

# According to the dimensionality reduction method of PCA of the training data of the model,
# the dimensionality reduction method of the new structure descriptor is guaranteed to remain unchanged
def trans(Data, model_path):
    with open(model_path, 'rb') as fw:
        model = pickle.load(fw)
        x_new= model.transform(Data)
        return x_new

# The first two dimensions are test frequency and temperature, followed by descriptors
data = data_all.values[:, 2:]

data_New=trans(data,model_path)

# Save the X after dimensionality reduction
data_pd = pd.DataFrame(data_New)


data_pd_out = pd.concat([data_pd_temp, data_pd], axis=1)

# Save the file, after this step the data has been completely processed and can be used for prediction
data_pd_out.to_csv('Test_Processing_completed_data.csv', index=False)