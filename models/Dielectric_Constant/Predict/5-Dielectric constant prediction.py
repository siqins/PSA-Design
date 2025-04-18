import pandas as pd
import numpy as np
import pickle
# The loading model is used for prediction
model_path=r'Di_17'
def prediction(X, model_path):
    with open(model_path, 'rb') as fw:
        model = pickle.load(fw)
        y_pre = model.predict(X)
        return y_pre

data=pd.read_csv('Test_Processing_completed_data.csv')
df=np.array(data.values[:,:])
c=df.reshape(-1,12)
e=prediction(c,model_path)
out_pd = pd.DataFrame(e)
out_pd.to_csv('Test_predict_data.csv', index=False)