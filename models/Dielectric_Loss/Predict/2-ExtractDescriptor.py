import pandas as pd

# Read raw data
data_all = pd.read_csv('Tset_all_descriptors.csv', low_memory=True)

# Get SMILES column and target variables
data_pd_temp = pd.read_csv('Tset_all_descriptors.csv', usecols=['RepeatUnitSMILES', 'TestFreq', 'TestTemp'])

# Select descriptor columns needed
data_all_1 = data_all.loc[:, ['SMR_VSA6', 'RotRatio', 'NssCH2', 'ATSC3Z', 'AATS3dv', 'BIC2', 'SLogP', 'fMF', 'nS',
                             'BalabanJ', 'C3SP2', 'AATSC1Z', 'nHBDon', 'NaaCH', 'MATS1Z', 'TMPC10', 'Xc-5d']]

# Combine SMILES and descriptors
data_pd_out = pd.concat([data_pd_temp, data_all_1], axis=1)

# Save file with SMILES column
data_pd_out.to_csv('Tset_SMILES_Y_X.csv', index=False)

# Create and save processed file without SMILES column
completed_data = data_pd_out.drop(columns=['RepeatUnitSMILES'])
completed_data.to_csv('Test_Processing_completed_data.csv', index=False)





























