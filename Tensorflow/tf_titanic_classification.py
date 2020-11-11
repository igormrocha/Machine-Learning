from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv').drop(['deck'],axis=1)
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv').drop(['deck'],axis=1) # training data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())

plt.figure()
dftrain.age.hist()

plt.figure()
dftrain.sex.value_counts().plot(kind='bar')

plt.figure()
pd.concat([dftrain,y_train],axis=1).groupby('sex').survived.mean().plot(kind = 'bar')

#plt.show()

dftrain = pd.get_dummies(dftrain, columns =['sex','embark_town','alone'])
dfeval = pd.get_dummies(dfeval, columns =['sex','embark_town','alone'])

print(dfeval.columns)

ORDINAL_COLUMNS = ['class']
NUMERIC_COLUMNS = dftrain.columns.drop(ORDINAL_COLUMNS)

print(NUMERIC_COLUMNS)

feature_columns = []
for feature_name in ORDINAL_COLUMNS:
	vocabulary = dftrain[feature_name].unique()
	feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
	feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
	def input_function():
		ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
		if shuffle:
			ds = ds.shuffle(1000)
		ds = ds.batch(batch_size).repeat(num_epochs)
		return ds
	return input_function

train_input_fn = make_input_fn(dftrain,y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs = 1, shuffle = False)

linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

#clear_output()
print(result)

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])
print(result[0])
			








