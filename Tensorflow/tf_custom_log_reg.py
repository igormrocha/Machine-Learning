from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np


dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv').drop(['deck'],axis=1)
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv').drop(['deck'],axis=1) # training data

dftrain = pd.get_dummies(dftrain, columns =['sex','embark_town','alone','class'],drop_first=True)

dfeval = pd.get_dummies(dfeval, columns =['sex','embark_town','alone','class'],drop_first=True)

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

x_train = np.array(dftrain, np.float32)
y_train = np.array(y_train, np.int32)
x_test = np.array(dfeval, np.float32)
y_test = np.array(y_eval, np.int32)


print(dftrain)

num_classes = 2 # Survived or not

num_features = len(dftrain.columns)

# Training parameters.

learning_rate = 0.001

training_steps = 1000

batch_size = 32

display_step = 10

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.repeat().shuffle(1000).batch(batch_size)

W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")

def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)

    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.

def accuracy(y_pred, y_true):

# Predicted class is the index of the highest score in prediction vector (i.e. argmax).

	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.

optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 

def run_optimization(x, y):

# Wrap computation inside a GradientTape for automatic differentiation.

    with tf.GradientTape() as g:

        pred = logistic_regression(x)

        loss = cross_entropy(pred, y)

    # Compute gradients.

    gradients = g.gradient(loss, [W, b])
 
    # Update W and b following gradients.

    optimizer.apply_gradients(zip(gradients, [W, b]))

# Run training for the given number of steps.

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    # Run the optimization to update W and b values.

    run_optimization(batch_x, batch_y)

    if step % display_step == 0:

        pred = logistic_regression(batch_x)

        loss = cross_entropy(pred, batch_y)

        acc = accuracy(pred, batch_y)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


# Test model on validation set.

pred = logistic_regression(x_test)

print("Test Accuracy: %f" % accuracy(pred, y_test))

pred = logistic_regression(x_test)

print("Test Accuracy: %f" % accuracy(pred, y_test))
 

