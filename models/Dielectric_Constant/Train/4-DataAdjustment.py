import pandas as pd

# Reading CSV file
df = pd.read_csv('Test_SMILES_Y_X.csv')

# Delete first column
df = df.drop(columns=['RepeatUnitSMILES'])

# Gets the data for the second column
property_value = df['PropertyValue']

# Delete the original second column
df = df.drop(columns=['PropertyValue'])

# Adds the original second column to the last column
df['PropertyValue'] = property_value

# Save the adjusted data to a new file
df.to_csv('Test_X_Y.csv', index=False)