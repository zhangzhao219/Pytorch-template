import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

data_train = pd.read_csv('./disaster2/train.csv',sep=',')
data_test = pd.read_csv('./disaster2/test.csv',sep=',')
del data_train['keyword']
del data_train['location']
del data_test['keyword']
del data_test['location']

for index,column in data_train.iterrows():
    if index % 1000 == 0:
        print(index)
    column['text'] = re.sub(r'http(.*)','http',string=column['text'])
    column['text'] = re.sub(r'[^a-zA-Z0-9\s]',' ',string=column['text'])
    column['text'] = [word for word in column['text'].split(' ') if word not in stop]
    column['text'] = ' '.join(column['text']).lower()
    column['text']=' '.join(column['text'].split())
    data_train.iat[index,0] = column['id']
    data_train.iat[index,1] = column['text']
    data_train.iat[index,2] = column['target']
for index,column in data_test.iterrows():
    if index % 1000 == 0:
        print(index)
    column['text'] = re.sub(r'http(.*)','http',string=column['text'])
    column['text'] = re.sub(r'[^a-zA-Z0-9\s]',' ',string=column['text'])
    column['text'] = [word for word in column['text'].split(' ') if word not in stop]
    column['text'] = ' '.join(column['text']).lower()
    column['text']=' '.join(column['text'].split())
    data_test.iat[index,0] = column['id']
    data_test.iat[index,1] = column['text']
data_test['target'] = ''
print(data_train['target'].value_counts())
data_train.to_csv('./disaster/train_data.csv',header=None,index=None)
data_test.to_csv('./disaster/test_data.csv',header=None,index=None)