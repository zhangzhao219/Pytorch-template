import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

data_train = pd.read_csv('./ag_news_csv_ori/train.csv',sep='","',names=['class','sentence1','sentence'])
data_test = pd.read_csv('./ag_news_csv_ori/test.csv',sep='","',names=['class','sentence1','sentence'])
del data_train['sentence1']
del data_test['sentence1']
for index,column in data_train.iterrows():
    if index % 1000 == 0:
        print(index)
    column['class'] = int(column['class'].split('"')[1]) - 1
    column['sentence'] = re.sub(r'[^a-zA-Z0-9\s]','',string=column['sentence'])
    column['sentence'] = [word for word in column['sentence'].split(' ') if word not in stop]
    column['sentence'] = ' '.join(column['sentence']).lower()
    column['sentence']=' '.join(column['sentence'].split())
for index,column in data_test.iterrows():
    if index % 1000 == 0:
        print(index)
    column['class'] = int(column['class'].split('"')[1]) - 1
    column['sentence'] = re.sub(r'[^a-zA-Z0-9\s]','',string=column['sentence'])
    column['sentence'] = [word for word in column['sentence'].split(' ') if word not in stop]
    column['sentence'] = ' '.join(column['sentence']).lower()
    column['sentence']=' '.join(column['sentence'].split())
print(data_test)
print(data_train['class'].value_counts())
print(data_test['class'].value_counts())
data_train.to_csv('./ag_news_csv/train_data.csv',header=None,index=None)
data_test.to_csv('./ag_news_csv/test_data.csv',header=None,index=None)