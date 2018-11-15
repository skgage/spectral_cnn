#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (batch_size, n, nrows, ncols, channels) where
    batchsize * n * nrows * ncols = arr.size

    If arr is a 4 array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    b, h, w, c = arr.shape
    print (b, h, w, c, nrows, ncols)
    return (arr.reshape(b, int(h/int(nrows)), int(nrows), -1, int(ncols), b)
               .swapaxes(1,2)
               .reshape(b, -1, nrows, ncols, c))
 
def blockshaped_transpose(arr):
    """
    Return an array of shape (batch_size, n, nrows, ncols, channels) where
    each n elements is tranposed as to produce E^T

    If arr is a 4D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    print ('element tensor shape = {}'.format(arr.shape))
    nblocks = arr.shape[1]
    nrows = arr.shape[2]
    ncols = arr.shape[3]
    transpose = blockshaped(np.reshape(np.transpose(arr, axes=(2,3,0,1,4)), 
                                       (-1,int(np.sqrt(nblocks)*nrows),int(np.sqrt(nblocks)*ncols),1)), 
                                        nrows, ncols)
    
    return transpose

def np_relu(x):
    return np.max([x,0])

def vec_np_relu(X):
    vec = np.vectorize(np_relu)
    return vec(X)

def tf_relu(X):
    return tf.nn.relu(X)

def kron_el(x, A):
    """ Input: Tensor x shape: [batch_size, 28, 28, 1]
               Tensor A shape: [nrows, ncols] w/ nrows and ncols = filter size

        Reshape tensor1 : [batch_size, nblocks, nchan, nrows, ncols]
        
        Computes E^T A E with tensordot op over axes=1
        
        Returns tensor of shape : [batch_size, nblocks, nrows, ncols, nchan]  """
    #Reshape X to 5-D tensor: [batch_size, num_elements, el_size, el_size, channels] using blockshaped
    nrows, ncols, ochan = A.shape
    E = blockshaped(x, nrows, ncols) #splits tensor into elements
    Et = blockshaped_transpose(E)
    batchsize, nblocks, nrows, ncols, nchan = E.shape
    E = tf.reshape(E, (-1, nblocks, int(nchan* nrows), ncols)) #for tensor product
    #batchsize, nblocks, nrows, ncols, nchan = Et.shape
    #A = np.reshape(A, (1,1,1,nrows,ncols))
    Et = tf.reshape(Et, [batchsize, nblocks, nchan, nrows, ncols])
    print ('Et shape = {}, A shape = {}'.format(np.shape(Et), np.shape(A)))
    EtA = tf.reshape(tf.tensordot(Et, A, axes=1), (-1, nblocks, int(int(ochan)*nrows), ncols))    #computes E^T A
    output = tf.reshape(tf.matmul(EtA, E), (-1, int(np.sqrt(nblocks)*nrows), int(np.sqrt(nblocks)*nrows), ochan)) #same output shape as EtA and E
    #Relu activation
    activation = tf_relu(output)
    return activation
    
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  #conv1 = tf.layers.conv2d(
      #inputs=input_layer,
      #filters=32,
      #kernel_size=[5, 5],
      #padding="same",
      #activation=tf.nn.relu)
  
   #REPLACEMENT CODE using tf.tensordot
   #Reshape X to 5-D tensor: [batch_size, num_elements, el_size, el_size, channels] using blockshaped
  fsize = 4 #filter size
  nfilters = 32
   #E = blockshaped(x, fsize, fsize) #splits tensor into elements
  A = tf.get_variable(name = 'A', shape=(fsize, fsize, nfilters), trainable=True)
  conv1 = kron_el(input_layer,A)
   #Filter Tensor A: [batch_size, 1, el_size, el_size, 1]; A is a trainable variable
   #need to transpose the elements in order to do op E^T A E
  
    
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  #For practice
  train_data = np.reshape(train_data[0,:], [1,784])
  train_labels = np.reshape(train_labels[0], [1,1])

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "softmax_tensor"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=2) #,
      #hooks=[logging_hook])

   #Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()