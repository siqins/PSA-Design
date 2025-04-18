import pandas as pd
# Replace the string and NaN value in the descriptor to be extracted from the predictive structure with 0

# Reading CSV file
df = pd.read_csv("Test_filter_descriptors.csv")

# Replace the string and NAN value
df = df.replace(to_replace='.*', value=0, regex=True)

df = df.fillna(0)

# Save CSV file
df.to_csv("Test_filter_descriptors.csv", index=False)