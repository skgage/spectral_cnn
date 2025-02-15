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
import math

tf.logging.set_verbosity(tf.logging.INFO)

def blockshaped(arr, nrows, ncols):
        """
        Return an array of shape (batch_size, n, nrows, ncols, channels) where
        batchsize * n * nrows * ncols = arr.size
        If arr is a 4 array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        b = tf.shape(arr)[0]
        h = tf.shape(arr)[1]
        w = tf.shape(arr)[2]
        c = tf.shape(arr)[3]
        print ((h*w)/(nrows*ncols))
        #print ('element tensor shape blockshaped = {}'.format(arr))
        output = tf.reshape(arr, (b, tf.cast((h*w)/(nrows*ncols), tf.int32), nrows, -1, ncols, c))
        #axes_switch tf.shape()
        output = tf.transpose(output, [0,2,1,3,4,5])
        output = tf.reshape(output, (b, -1, nrows, ncols, c))
        #print ('output shape of blockshpaed = {}'.format(output))
        return output
                             
 
def blockshaped_transpose(arr):
        """
        Return an array of shape (batch_size, n, nrows, ncols, channels) where
        each n elements is tranposnied as to produce E^T
        If arr is a 4D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        #print ('element tensor shape b_trans = {}'.format(arr))
        nblocks = tf.cast(tf.shape(arr)[1], tf.float32)
        #if nblocks is None:
                #print ("clearly is nonetype")
                #nblocks = int(-1)
        #else:
                #print ("thinks it is nonetype")
             # nblocks = int(nblocks)
        nrows = tf.shape(arr)[2]
        ncols = tf.shape(arr)[3]
        transpose = blockshaped(tf.reshape(tf.transpose(arr, [2,3,0,1,4]), 
                                                                             (-1, tf.cast(tf.sqrt(nblocks), dtype=tf.int32)*nrows,
                                                                                tf.cast(tf.sqrt(nblocks), dtype=tf.int32)*ncols,1)),
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
        """ Input: Tensor x shape: [batch_size, nblocks, nrows, ncols, nchans]
            Input: Tensor x shape: [batch_size, 28, 28, 1]
                             Tensor A shape: [nrows, ncols] w/ nrows and ncols = filter size
                Reshape tensor1 : [batch_size, nblocks, nchan, nrows, ncols]
                
                Computes E^T A E with tensordot op over axes=1
                
                Returns tensor of shape : [batch_size, nblocks, nrows, ncols, nchan]  """
        #Reshape X to 5-D tensor: [batch_size, num_elements, el_size, el_size, channels] using blockshaped
        #nrows, ncols, ochan = A.shape
        E = x
        nrows = tf.shape(A)[0]
        ncols = tf.shape(A)[1]
        ochan = tf.shape(A)[2]
        if tf.shape(x)[1] == 28:
            E = blockshaped(x, nrows, ncols) #splits tensor into elements
        Et = blockshaped_transpose(E)
        #batchsize, nblocks, nrows, ncols, nchan = E.shape
        batchsize = tf.shape(E)[0]
        nblocks = tf.shape(E)[1]
        nrows = tf.shape(E)[2]
        ncols = tf.shape(E)[3]
        nchan = tf.shape(E)[4]
        E = tf.reshape(E, (-1, nblocks, nrows*nchan, ncols)) #for tensor product
        #batchsize, nblocks, nrows, ncols, nchan = Et.shape
        #A = np.reshape(A, (1,1,1,nrows,ncols))
        Et = tf.reshape(Et, [batchsize, nblocks, nchan, nrows, ncols])
        EtA = tf.tensordot(Et, A, axes=1)
        print ('Et shape = {}, A shape = {}'.format(np.shape(Et), np.shape(A)))
        print ('EtA shape = {}, E shape = {}'.format(np.shape(EtA), np.shape(E)))
        EtA = tf.reshape(EtA, (-1, nblocks, nrows* ochan, ncols*nchan))  #computes E^T A
        print ('EtA shape = {} after reshape'.format(np.shape(EtA)))
        output = tf.matmul(EtA, E)
        print ('output shape = {}'.format(np.shape(output)))
        output = tf.reshape(output, (-1, 
                                                tf.cast(tf.sqrt(tf.cast(nblocks, tf.float32)), tf.int32)*nrows,
                                                tf.cast(tf.sqrt(tf.cast(nblocks, tf.float32)), tf.int32)*ncols,
                                                ochan)) #same output shape as EtA and E
        output = tf.reshape(output, (output.shape[0], nblocks,ochan, 4, tf.cast(nrows/2, tf.int32),tf.cast(ncols/2, tf.int32)))
        #Relu activation
        activation = tf_relu(output)
        print ('activation shape = {}'.format(np.shape(activation)))
        #activation = tf.reshape(activation, (batchsize, nblocks, nrows, ncols, ochan))
        return activation

def pooling(inputs):
    batchsize = inputs.shape[0] 
    nblocks = inputs.shape[1]
    print ('nblocks = {}'.format(nblocks))
    nrows = inputs.shape[2]
    ncols = inputs.shape[3]
    ochan = inputs.shape[4]
    print ('batchsize = {}'.format(batchsize))
    n_splits = 4 #tf.sqrt(nrows)*tf.sqrt(ncols)
    #row_split = tf.split(inputs, num_or_size_splits=n_splits, axis=2) #split along rows
    #print (tf.shape(row_split))

    #reshaped to take transfer subblocks of elements to reduce, ie higher order element to lower
    inputs = tf.reshape(inputs, [batchsize, nblocks, n_splits, 2, 2, ochan])
    inputs_pooled = []
    for i in range(batchsize):      #for each mesh in the batch
        for j in range(nblocks):        #for each element in the individual mesh
            #based on element dimensions, split element into subelements of size sqrt nrows
            #n_splits = 2
            #row_splits = tf.split(inputs[i,j,:,:,:], num_or_size_splits=n_splits, axis=0)
            #col_splits = tf.split(row_splits,num_or_size_splits=n_splits, axis=0)
            #print (col_splits[0])

            #hard coded splits for 2D quads - NEED TO GENERALIZE
            #split1 = inputs[i,j,:2,:2,:] #upper left
            #split2 = inputs[i,j,:2,2:,:] #upper right
            #split3 = inputs[i,j,2:,:2,:] #lower left
            #split4 = inputs[i,j,2:,2:,:] #lower right   
            #print (split1, split2, split3, split4)
            #sess = tf.Session()
            #with sess.as_default():
                #tf.initialize_all_variables().run()
            for k in range(ochan):
                for l in range(n_splits):
                    mean_element = tf.reduce_mean(inputs[i,j,l,:,:,k], axis=0)[0]
                    inputs_pooled = tf.concat([inputs_pooled, [mean_element]], axis=0)
                    #print (mean_element)
                    #inputs_pooled[i,j,l,k] = mean_element
    inputs_pooled = tf.reshape(inputs_pooled, [batchsize, nblocks, 2,2,ochan])
    print (inputs_pooled)
    return inputs_pooled

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
#  conv1 = tf.layers.conv2d(
#      inputs=input_layer,
#      filters=32,
#      kernel_size=[5, 5],
#      padding="same",
#      activation=tf.nn.relu)
    
     #REPLACEMENT CODE using tf.tensordot
     #Reshape X to 5-D tensor: [batch_size, num_elements, el_size, el_size, channels] using blockshaped
    fsize = 4 #filter size
    nfilters = 32
     #E = blockshaped(x, fsize, fsize) #splits tensor into elements
    A = tf.get_variable(name = 'A', shape=(fsize, fsize, nfilters), trainable=True)
    input_layer = tf.reshape(input_layer, (input_layer.shape[0], -1,fsize,fsize, 1))
    conv1 = kron_el(input_layer,A)
    #temp_reshape_conv1 = tf.reshape(conv1, (conv1.shape[0],28,28,nfilters))
    #temp_reshape_conv1 = tf.reshape(conv1, (conv1.shape[0], 49,nfilters,4, 2,2))
    #print (temp_reshape_conv1)
     #Filter Tensor A: [batch_size, 1, el_size, el_size, 1]; A is a trainable variable
     #need to transpose the elements in order to do op E^T A E
    x1 = np.reshape(np.array([0.5,0.5], np.float32), (1,2))
    x2 = np.reshape(x1, (2,1))
    pool1 = tf.tensordot(tf.reduce_mean(conv1, axis=4), x2, axes=1)
    print (pool1) #shape should be [1,49,32,4,1]
    pool1 = tf.reshape(pool1, [pool1.shape[0],49,2,2,32]) #temp
    print (pool1)
        
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #pool1 = pooling(conv1)
    #temp_reshape_pool1 = tf.reshape(pool1, [pool1.shape[0], 14,14,32])
    #print (temp_reshape_pool1)
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    '''conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)'''

    #Replacement for conv2
    fsize = 2 #LINEAR ELEMENT SIZE
    nfilters = 64
    B = tf.get_variable(name='B', shape=(fsize, fsize, nfilters), trainable=True)
    conv2 = kron_el(pool1, B)
    print ('conv2 = {}'.format(conv2))
    #temp_reshape_conv2 = tf.reshape(conv2, (conv2.shape[0],14,14,nfilters))
    
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    #pool2 = tf.layers.max_pooling2d(inputs=temp_reshape_conv2, pool_size=[2, 2], strides=2)

    #x1 = np.reshape(np.array([0.5,0.5], np.float32), (1,2))
    #x2 = np.reshape(x1, (2,1))
    pool2 = tf.reduce_mean(conv2, axis=3)
    print ('pool2 = {}'.format(pool2)) #shape should be [1,49,32,4,1]
    #pool2 = tf.reshape(pool1, [pool.shape[0],49,2,2,32]) #temp
    #print (pool1)
    #pool2 = pooling(conv2)
    #temp_reshape_pool1 = tf.reshape(pool2, [pool1.shape[0], 7,7,64])
    #print (temp_reshape_pool2)
    
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
    #tf.reset_default_graph()
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    #For practice
    #train_data = np.reshape(train_data[:1,:], [1,784])
    #train_labels = np.reshape(train_labels[:1], [1,1])

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir ="")#/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "softmax_tensor"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=10,
            num_epochs=None,
            shuffle=True)
    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000) #,
            #hooks=[logging_hook])

     #Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

#main()

if __name__ == "__main__":
    tf.app.run()
