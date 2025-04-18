import pandas as pd

# Reading CSV file
df = pd.read_csv('Test_SMILES_Y_X.csv')

# Delete first column
df = df.drop(columns=['RepeatUnitSMILES'])

# Gets the data for the second column
td_value = df['td_value']

# Delete the original second column
df = df.drop(columns=['td_value'])

# Adds the original second column to the last column
df['td_value'] = td_value

# Save the adjusted data to a new file
df.to_csv('Test_X_Y.csv', index=False)