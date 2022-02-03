import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path


###################Reading Records#############
data_filename=str(Path(__file__).parent / 'data/datalocation.txt')
with open(data_filename) as f:
    df_name = f.readlines()

trainingdata=pd.read_csv(df_name)


##################Re-training a Model#############
X=trainingdata.loc[:,['col1','col2']].values.reshape(-1, 2)
y=trainingdata['col3'].values.reshape(-1, 1)

logit=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

logit.fit(X,y)
                    




############Pushing to Production###################
model_filename=str(Path(__file__).parent / 'model/deployedmodelname.txt')
with open(model_filename) as f:
    model_name = f.readlines()

pickle.dump(logit,model_name)






