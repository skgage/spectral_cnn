import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import inc_res
import cv2

h = 1 #TEMPORARY
def T(num,h):
    if num == 1:
        T = np.array([[1,0,0,0,0,1/(4*h**2),1/(2*h**3),1/(2*h**3),1/(h**4)],
            [1,0,1/(2*h),0,1/(h**2),0,-2/(2*h**3),0,-2/(h**4)],
            [1,0,0,0,0,-1/(4*h**2),1/(2*h**3),-1/(2*h**3),1/(h**4)],
            [1,1/(2*h),0,1/(h**2),0,0,0,-2/(2*h**3),-2/(h**4)],
            [1,0,0,-2/(h**2),-2/(h**2),0,0,0,4/(h**4)],
            [1,-1/(2*h),0,1/(h**2),0,0,0,2/(2*h**3),-2/(h**4)],
            [1,0,0,0,0,-1/(4*h**2),-1/(2*h**3),1/(2*h**3),1/(h**4)],
            [1,0,-1/(2*h),0,1/(h**2),0,2/(2*h**3),0,-2/(h**4)],
            [1,0,0,0,0,1/(4*h**2),-1/(2*h**3),-1/(2*h**3),1/(h**4)]])
    elif num == 2:
        T = np.array([[0,0,0,0,0,1/(4*h**2),1/(2*h**3),1/(2*h**3),1/(h**4)],
            [0,0,1,0,1/(h**2),0,-2/(2*h**3),0,-2/(h**4)],
            [0,0,1,0,0,-1,1/(2*h**3),-1/(2*h**3),1/(h**4)],
            [0,1,0,1/(h**2),0,0,0,-2/(2*h**3),-2/(h**4)],
            [1,0,0,-2/(h**2),-2/(h**2),0,0,0,4/(h**4)],
            [1,-1,0,1/(h**2),0,0,0,2/(2*h**3),-2/(h**4)],
            [0,1,0,0,0,-1,-1/(2*h**3),1/(2*h**3),1/(h**4)],
            [1,0,-1,0,1/(h**2),0,2/(2*h**3),0,-2/(h**4)],
            [1,-1,-1,0,0,1,-1/(2*h**3),-1/(2*h**3),1/(h**4)]])
    else:
        print ('{} not an option'.format(num))
    return T
#print ('T = ', T)
'''
with open('allvars_file.pkl', 'rb') as f:
    #a = pickle.load(f)
    #print ('a ',a)
    [weights1i, bias1i, weights2i, bias2i, wd1i, bd1i, wd2i, bd2i] = pickle.load(f)
    print (weights1i.shape,bias1i.shape, weights2i.shape,bias2i.shape, wd1i.shape, bd1i.shape,wd2i.shape, bd2i.shape)

#with open('weights_file.pkl', 'rb') as f:
    #weights1, weights2 = pickle.load(f)
    #print ('weights1 = ', weights1.shape, " weights2 = ", weights2.shape)
K1 = np.linalg.inv(T(1,1))@np.reshape(weights1i, [9,32])
K2 = np.linalg.inv(T(1,1))@np.reshape(weights2i, [9,2048])
check_w1 = np.matmul(T(1,1),K1) #T@K1
print ('check_w1 = ', check_w1)
if (check_w1.all() == weights1i.all()):
    print ('Yay2!')
else:
    print ('Nah2')
check_w2 = T(1,1)@K2
if (check_w2.all() == weights2i.all()):
    print ('Yay2!')
else:
    print ('Nah2')
    '''
def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Python optimisation variables
    learning_rate = 0.001
    epochs = 100
    batch_size = 50

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()
    #WHAT IF HIGHER RESOLUTION IMAGE WERE CREATED HERE/DON'T WANT TO DO EVERYTIME
    x = tf.placeholder(tf.float32, [None, 784])    # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
    x_highres = tf.placeholder(tf.float32, [None, 3136])
    # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
    # x 28).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
    # dimension would be 3
    print ('x shape: ', x.shape, ' x_highres shape: ', x_highres.shape)
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    x_highres_shaped = tf.reshape(x_highres, [-1, 56, 56, 1])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # create some convolutional layers
    layer1, weights1, bias1 = create_new_conv_layer(x_shaped, 1, 32, [3, 3], [2, 2], name='layer1', trainable=True)
    layer2, weights2, bias2 = create_new_conv_layer(layer1, 32, 64, [3, 3], [2, 2], name='layer2', trainable=True)
    K1 = np.linalg.inv(T(1,1))@tf.reshape(weights1, [9,32])
    K2 = np.linalg.inv(T(1,1))@tf.reshape(weights2, [9,2048])
    print ('conv layer 2 output OG: ', layer2.shape, tf.size(layer2))
    print('K1: ', K1.shape, ' K2: ', K2.shape)
    layer1_highres, _, _ = create_new_conv_layer(x_highres_shaped, 1, 32, [3, 3], [2, 2], name='layer1_highres',init_w=tf.reshape(T(1,2)@K1, [3,3,1,32]),init_b = bias1, trainable=False)
    print ('conv layer 1 output high res: ', layer1_highres.shape, tf.size(layer1_highres))
    layer2_highres, _, _ = create_new_conv_layer(layer1_highres, 32, 64, [3, 3], [2, 2], name='layer2_highres',init_w=tf.reshape(T(1,2)@K2, [3,3,32,64]), init_b = bias2, trainable=False)
    
    print ('conv layer 2 output high res: ', layer2_highres.shape, tf.size(layer2_highres))
    #This additional pooling layer is for the higher resolution images
    layer2_highres = tf.nn.max_pool(layer2_highres, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
    # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
    # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
    flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
    flattened_highres = tf.reshape(layer2_highres, [-1, 7 * 7 * 64])
    # setup some weights and bias values for this layer, then activate with ReLU
    wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    #testing
    #wd1 = tf.Variable(wd1i,name='wd1')
    #bd1 = tf.Variable(bd1i, name='bd1')

    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    dense_layer1_highres = tf.matmul(flattened_highres, wd1) + bd1
    dense_layer1_highres = tf.nn.relu(dense_layer1_highres)

    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    #testing
    #wd2 = tf.Variable(wd2i,name='wd2')
    #bd2 = tf.Variable(bd2i, name='bd2')

    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    dense_layer2_highres = tf.matmul(dense_layer1_highres, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2) #want this output for 
    y_highres = tf.nn.softmax(dense_layer2_highres)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
    cross_entropy_highres = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2_highres, labels=y))
    total_cost = cross_entropy + cross_entropy_highres
    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction_highres = tf.equal(tf.argmax(y, 1), tf.argmax(y_highres, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) + tf.reduce_mean(tf.cast(correct_prediction_highres, tf.float32)))/2
    accuracy_OG = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_highres = tf.reduce_mean(tf.cast(correct_prediction_highres, tf.float32))
    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy
    #tf.summary.scalar('accuracy', accuracy)

    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
    with tf.Session() as sess:
        # initialise the variables
        print ('size of dataset: ', mnist.train.images.shape)
        rand_choice = np.random.randint(mnist.train.images.shape[0], size=int(len(mnist.train.images)/10))
        train_images = mnist.train.images[rand_choice, :]
        train_labels = mnist.train.labels[rand_choice, :]
        train_images_highres = inc_res.increase_imresolution(mnist.train.images[rand_choice, :], 2, 'bilinear')
        train_images_lowres = inc_res.increase_imresolution(mnist.train.images[rand_choice, :], 1/2, 'bilinear')
        print ('OG shape {} High res shape {} Low res shape {}'.format(train_images.shape, train_images_highres.shape, train_images_lowres.shape))
        print ('reduced quantity of training data: ', train_images.shape)
        '''train_allres = np.column_stack((np.column_stack((train_images.T, train_images_highres.T)),train_images_lowres.T))
        print ('Stacked training data all res {}'.format(train_allres.shape))
        ex_image = np.reshape(train_allres[0,:], [28,28])
        cv2.imshow('image',ex_image)
        cv2.waitKey(0)
        ex_image = train_allres[0,:]
        cv2.imshow('image',ex_image)
        cv2.waitKey(0) 
        ex_image = train_allres[0,:]
        cv2.imshow('image',ex_image)
        cv2.waitKey(0)       '''
        sess.run(init_op)
        total_batch = int(len(train_labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                #print ('batch ', i, ' goes from index ', i*batch_size, ' to ', (i+1)*batch_size-1)
                #print (train_images[i*batch_size:(i+1)*batch_size, :].shape)
                batch_x = train_images[i*batch_size:((i+1)*batch_size), :] #next_batch(batch_size=batch_size)
                batch_y = train_labels[i*batch_size:((i+1)*batch_size), :]
                batch_x_highres = train_images_highres[i*batch_size:((i+1)*batch_size), :]                #print ('batch shape: {} and num values: {}'.format(batch_x.shape, np.size(batch_x)))
                _, c = sess.run([optimiser, total_cost], feed_dict={x: batch_x, x_highres: batch_x_highres, y: batch_y})
                avg_cost += c / total_batch
            var_list = []
            '''
            for var in tf.trainable_variables():
                var_val = sess.run(var)
                var_list.append(var_val)
                #print(var.name)
                if var.name == 'layer1_W:0':
                    weights1 = var_val
                    #print('Yay, {}, {}'.format(var.name, var_val.shape))
                elif var.name == 'layer2_W:0':
                    weights2 = var_val
                    #print('Yay, {}, {}'.format(var.name, var_val.shape))
                else:
                    continue
                    '''
            '''K1 = np.linalg.inv(T)@np.reshape(weights1, [9,32])
            K2 = np.linalg.inv(T)@np.reshape(weights2, [9,2048])'''

            #Increased resolution images
            #When need x times increased resoluton version, use code below 
            #interp can be 'bilinear', 'bicubic', and 'nearest'
            train_acc = sess.run(accuracy_OG, feed_dict={x: train_images, x_highres: train_images_highres, y: train_labels})
            test_acc = sess.run(accuracy_OG, feed_dict={x: mnist.test.images, x_highres: inc_res.increase_imresolution(mnist.test.images,2,'bilinear'), y: mnist.test.labels})
            train_acc_highres = sess.run(accuracy_highres, feed_dict={x: train_images, x_highres: train_images_highres, y: train_labels})
            test_acc_highres = sess.run(accuracy_highres, feed_dict={x: mnist.test.images, x_highres: inc_res.increase_imresolution(mnist.test.images,2,'bilinear'), y: mnist.test.labels})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " train accuracy OG: {:.3f}".format(train_acc), " test accuracy OG: {:.3f}".format(test_acc))
            print(" train accuracy high res: ", train_acc_highres, ' test accuracy highres: ', test_acc_highres)
            #summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            #writer.add_summary(summary, epoch)

        #with open('weights_file.pkl','wb') as file:
            #pickle.dump([weights1, weights2], file)
        #with open('allvars_file.pkl','wb') as file:
            #pickle.dump(var_list, file)
        '''
        interp = 'bilinear'
        factor = 2
        eval_images = inc_res.increase_imresolution(factor,interp)
        print ('eval_images = ', eval_images.shape)

        print("\nTraining complete!")
        writer.add_graph(sess.graph)
        print("\nTesting Accuracy")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        #print(sess.run(accuracy, feed_dict={x: eval_images, y: mnist.test.labels}))
        '''

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name,init_w=None,init_b=None, trainable=True):
    #layer1 = create_new_conv_layer(x_shaped, 1, 32, [3, 3], [2, 2], name='layer1')
    #layer2 = create_new_conv_layer(layer1, 32, 64, [3, 3], [2, 2], name='layer2')
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    #weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    #bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    if init_w == None and init_b == None:
        init_w = tf.truncated_normal(conv_filt_shape, stddev=0.03)
        init_b = tf.truncated_normal([num_filters])
    #for testing uncomment these
    weights = tf.Variable(init_w, name=name+'_W',dtype=tf.float32, trainable=trainable)
    bias = tf.Variable(init_b, name=name+'_b', dtype=tf.float32, trainable=trainable)
    

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
    # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
    # applied to each channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
    # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
    # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
    # to do strides of 2 in the x and y directions.
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return [out_layer, weights, bias]

if __name__ == "__main__":
    run_cnn()
