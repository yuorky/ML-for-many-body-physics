############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 1 ############
### Code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code builds a simple data set of spirals with K branches and then implements
### and trains a simple feedforward neural network to classify its branches.
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)

plt.ion() # turn on interactive mode (for plotting)

############################################################################
####################### CREATE AND PLOT THE DATA SET #######################
############################################################################
x_data = np.loadtxt('x_L30.txt',dtype='int8')
y_data = np.loadtxt('y_L30.txt',dtype='int8')
T_data = np.loadtxt('T_L30.txt',dtype='int8')

X_train, X_test, y_train, y_test = train_test_split(
        x_data,y_data,test_size=0.1)

X_train, X_val, y_train, y_val = train_test_split(
        X_train,y_train,test_size=0.2222)

N = 50 # number of points per branch
K = 2  # number of branches



############################################################################
##################### DEFINE THE NETWORK ARCHITECTURE ######################
############################################################################

### Create placeholders for the input data and labels ###
### (we'll input actual values when we ask TensorFlow to run an actual computation later) ###
x = tf.placeholder(tf.float32, [None, 900]) # input data
y = tf.placeholder(tf.int32,[None])       # labels

### Layer 1: ###
W1 = tf.Variable( tf.random_normal([900, 100], mean=0.0, stddev=0.01, dtype=tf.float32) )
b1 = tf.Variable( tf.zeros([100]) )
z1 = tf.matmul(x, W1) + b1
a1 = tf.nn.sigmoid( z1 )

W2 = W1 = tf.Variable( tf.random_normal([100, 2], mean=0.0, stddev=0.01, dtype=tf.float32) )
b2 = tf.Variable( tf.zeros([2]) )
z2 = tf.matmul(a1,W2) + b2
a2 = tf.nn.sigmoid(z2)
### Network output: ###
aL = a2

### Cost function: ###
### (measures how far off our model is from the labels) ###
y_onehot = tf.one_hot(y,depth=K) # labels are converted to one-hot representation
eps=0.0000000001 # to prevent the logs from diverging
cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_onehot * tf.log(aL+eps) +  (1.0-y_onehot )*tf.log(1.0-aL +eps) , reduction_indices=[1]))
cost_func = cross_entropy

### Use backpropagation to minimize the cost function using the gradient descent algorithm: ###
learning_rate  = 1.0 # hyperparameter
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

N_epochs = 20000 # number of times to run gradient descent

##############################################################################
################################## TRAINING ##################################
##############################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch_list    = []
cost_training = []
acc_training  = []

############ Function for plotting: ############
def updatePlot():

    ### Generate coordinates covering the whole plane: ###
#    padding = 0.1
#    spacing = 0.02
#    x1_min, x1_max = x_train[:, 0].min() - padding, x_train[:, 0].max() + padding
#    x2_min, x2_max = x_train[:, 1].min() - padding, x_train[:, 1].max() + padding
#    x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, spacing),
#                         np.arange(x2_min, x2_max, spacing))

    NN_output       = sess.run(aL,feed_dict={x:np.c_[x1_grid.ravel(), x2_grid.ravel()]})
    predicted_class = np.argmax(NN_output, axis=1)

    ### Plot the classifier: ###
    plt.subplot(121)
    plt.contourf(x1_grid, x2_grid, predicted_class.reshape(x1_grid.shape), K, alpha=0.8)
#    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)
    plt.xlim(x1_grid.min(), x1_grid.max())
    plt.ylim(x2_grid.min(), x2_grid.max())
    plt.xlabel('x1')
    plt.ylabel('x2')

    ### Plot the cost function during training: ###
    plt.subplot(222)
    plt.plot(epoch_list,cost_training,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Training cost')

    ### Plot the training accuracy: ###
    plt.subplot(224)
    plt.plot(epoch_list,acc_training,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
############ End of plotting function ############

### Train for several epochs: ###
for epoch in range(N_epochs):
    sess.run(train_step, feed_dict={x: X_train,y:y_train}) #run gradient descent
    
    ### Update the plot and print results every 500 epochs: ###
    if epoch % 500 == 0:
        cost = sess.run(cost_func,feed_dict={x:X_train, y:y_train})
        NN_output = sess.run(aL,feed_dict={x:X_train, y:y_train})
        predicted_class = np.argmax(NN_output, axis=1)
        accuracy = np.mean(predicted_class == y_train)
    
        print( "Iteration %d:\n  Training cost %f\n  Training accuracy %f\n" % (epoch, cost, accuracy) )
    
        epoch_list.append(epoch)
        cost_training.append(cost)
        acc_training.append(accuracy)
        
        ### Update the plot of the resulting classifier: ###
        fig = plt.figure(2,figsize=(10,5))
        fig.subplots_adjust(hspace=.3,wspace=.3)
        plt.clf()
        updatePlot()
        plt.pause(0.1)

plt.savefig('spiral_results.pdf') # Save the figure showing the results in the current directory

plt.show()
