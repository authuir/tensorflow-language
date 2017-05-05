import sys
from dataset import DataSet_English
from tensorflow.python.framework import dtypes

print(DataSet_English.train_file)
print(DataSet_English.train_size)

data_train, label_train = DataSet_English.data_from_text(DataSet_English.train_file,DataSet_English.train_size)
train = DataSet_English(data_train, label_train, dtype=dtypes.float32)
data_train, label_train = DataSet_English.data_from_text(DataSet_English.train_file,DataSet_English.train_size)
train = DataSet_English(data_train, label_train, dtype=dtypes.float32)