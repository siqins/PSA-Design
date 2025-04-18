import pandas as pd
import numpy as np

import pickle

model_path=r'Di_0'

def prediction(X, model_path):
    with open(model_path, 'rb') as fw:
        model = pickle.load(fw)
        y_pre = model.predict(X)
        return y_pre

data=pd.read_csv('Test_Processing_completed_data.csv')
df=np.array(data)
c=df.reshape(-1,19)
e=prediction(c,model_path)
out_pd = pd.DataFrame(e)
outName = 'Test_predict_data' + '.csv'
out_pd.to_csv(outName, index=False)
