import pandas as pd

# In all descriptors of the predictive structure, the descriptors that existed
# before dimensionality reduction of the training model data and the test frequency and temperature
# were extracted, and the data after dimensionality reduction of these descriptors plus frequency and temperature
# were used as the input of the model
descriptor_use=pd.read_csv("Td5_required_descriptor.csv",nrows=1)
descriptor_temp = pd.read_csv('Tset_all_descriptors.csv',usecols=descriptor_use)
descriptor=pd.DataFrame(descriptor_temp)
descriptor.to_csv('Test_filter_descriptors.csv',index=False)