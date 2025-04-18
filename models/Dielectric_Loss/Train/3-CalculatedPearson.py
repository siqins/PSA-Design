import pandas as pd
data=pd.read_csv('Test_filter_descriptors.csv')
df=pd.DataFrame(data)
df.corr()
out_pd = pd.DataFrame(df.corr())
outName = 'Test_corr(pearson)' + '.csv'
out_pd.to_csv(outName, index=False)


