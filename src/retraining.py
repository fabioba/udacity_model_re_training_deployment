import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path


###################Reading Records#############
data_filename=str(Path(__file__).parent.parent / 'data/datalocation.txt')
with open(data_filename) as f:
    df_name = f.read()
print(df_name)

data_filename=str(Path(__file__).parent.parent / 'data' / df_name)

trainingdata=pd.read_csv(data_filename)


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
model_filename=str(Path(__file__).parent.parent / 'model/deployedmodelname.txt')
with open(model_filename) as f:
    model_name = f.read()

model_filename=str(Path(__file__).parent.parent / 'model' / model_name)
pickle.dump(logit,open(model_filename,'wb'))






