'''
Demo of unsupervised learning of tensor structures.

project repo: https://github.com/GeometryOfData/2017_04_DeepTensor

@author: Roy Lederman
'''

import random

import numpy as np

import tensorflow as tf

import deepquest_aux as dqax

#visualization and reading image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
pltcnt = 1;

from sklearn.manifold import TSNE
import DiffMaps as diffmaps

#
# Run name
#
myrunstr = 'test_002'


# Optimization parameters
#

#number of iterations
num_iter = 10000 

# TF learning rate exponential decay
learning_rate_start = 0.05
learning_rate_global_step = 1000
learning_rate_decay_rate = 0.98

# Number of ADAM optimization step, before switching to SGD.
# ADAM works better at least in the beginning, so it is good to at least start with it.
# Negative number for no ADAM.
optimization_how_many_adam = 2000  


#
# Visualization parameters
#
DM_numneigh = 20
DM_islocal = 1
show_results_once_in = 2000
show_scores_once_in = 2000

imshow_dims = 3

#
# Generating the dataset
#

testdat = matplotlib.image.imread('Lenna.png')

# data dimensionality 
La=testdat.shape[0]
Lb=testdat.shape[1]
Ly=testdat.shape[2]

print('reformat')
datx, daty = dqax.reformatdat(testdat)

plt.imshow(testdat[:,:,0:imshow_dims]) ; 
plt.title('The first coordinate of the entries in the dataset (each entry is a vector)');
plt.show()

#
# Network parameters
#

batch_size =  np.int(np.sqrt(La*Lb/2) * 5)
embedding_size_a = 15;#20;#8; 45; 
embedding_size_b = 15;#20;#8; 45; 

layer1_size = 30;#10; 25
layer2_size = 30;#5;10; 25
layer3_size = 30;#5;10; 25


print('Batch size: '+str(batch_size))

#
#
# Test and train sets
#
np.random.seed(123) # Data randomization seed

training_fraction = 0.2

dataperm_ids = np.random.permutation(La*Lb)
# ids to use in the training set: the first half of the permuted list of indices
dataperm_ids_train = dataperm_ids[ range(np.int(La*Lb * training_fraction) ) ] 
# ids to use in the test set: the second half of the permuted list of indices
dataperm_ids_test  = dataperm_ids[ range(np.int(La*Lb * training_fraction),La*Lb ) ] 

# create the actual sets:
train_y = daty[dataperm_ids_train,:]
test_y = daty[dataperm_ids_test,:]
train_x = datx[dataperm_ids_train,:]
test_x = datx[dataperm_ids_test,:]


#
# Show the training set (first coordinate only)
#
dat_sp = np.zeros([La,Lb,Ly])
for j1 in range(len(train_x)):
    dat_sp[np.int(train_x[j1,0]), np.int(train_x[j1,1])] =train_y[j1]
#plt.imshow(dat_sp[:,:,0:3])
#plt.show()
plt.imshow(dat_sp[:,:,0:imshow_dims])
plt.title('The first coordinate of the entries in the sampled dataset (with missing entries)');
plt.show()



#
# Define the network model
#


sess = tf.InteractiveSession()

# Input variables:
# x is the two coordinates (indices) of in the 2-D matrix
x = tf.placeholder(tf.int32, shape=[None, 2]) 
# y is the entry at each coordinate
y_ = tf.placeholder(tf.float32, shape=[None, Ly])
# gstep is used to compute the step size (it would usually be the number of the iteration).
# it is  a technical variable used in the optimization
gstep = tf.placeholder(tf.int32)


#
# Embedding layer 0
#

#
# these are matrices of La (or Lb) rows, each with embedding_size columns,
# each column is the vector in \mathbb{R}^embedding_size .
# 
# These embeddings are variables that the net optimizes. 
#
embeddings_a = tf.Variable(
         tf.random_uniform([La, embedding_size_a], -1.0, 1.0))
embeddings_b = tf.Variable(
         tf.random_uniform([Lb, embedding_size_b], -1.0, 1.0))
#embeddings_a = tf.Variable(
#        tf.truncated_normal([La, embedding_size_a],stddev=0.1))
#embeddings_b = tf.Variable(
#        tf.truncated_normal([Lb, embedding_size_b],stddev=0.1))

#
# Id to vector
#
# First step is to take the index of the first dimension and turn it into a vector,
# and to take the index of the second dimension and turn it into a vector.
embed_a = tf.gather(embeddings_a, x[:,0])
embed_b = tf.gather(embeddings_b, x[:,1])


#
# Layer 1
#

#
# Layer 1 combines the two representations:
# F1( embd_a , embd_b ) = Relu ( Wa * embd_a + Wb * embd_b + b )
# where Wa, Wb and b are variables in the optimizations. 
#

# variables of layer one
Wa = tf.Variable(tf.truncated_normal([embedding_size_a, layer1_size],stddev=0.02)) 
Wb = tf.Variable(tf.truncated_normal([embedding_size_b, layer1_size],stddev=0.02))
F1b = tf.Variable(tf.random_uniform([layer1_size],-0.05,0.05))

# the operation in Layer 1
F1 = tf.nn.relu( tf.matmul( embed_a, Wa) + tf.matmul( embed_b, Wb) + F1b ); 

#
# Layer 2
#

# Layer 2 is a standard layer of a neural network
W2 = tf.Variable(tf.truncated_normal([layer1_size, layer2_size],stddev=0.1))
F2b = tf.Variable(tf.random_uniform([layer2_size],-1.0,1.0))

F2 = tf.nn.relu( tf.matmul( F1, W2) + F2b );

#
# Layer 3
#

# another standard layer

W3 = tf.Variable(tf.truncated_normal([layer2_size, layer3_size],stddev=0.1))
F3b = tf.Variable(tf.random_uniform([layer3_size],-1.0,1.0))

F3 = tf.nn.relu( tf.matmul( F2, W3) + F3b );


#
# Output layer 
#

# output is of size Ly, the size of the entries

FOW = tf.Variable(tf.truncated_normal([layer3_size, Ly],stddev=0.1))
#FOb = tf.Variable(tf.random_uniform([Ly],0.0,3.0))
FOb = tf.Variable(tf.random_uniform([Ly],0.0,3.0))

FO = tf.matmul(F3, FOW) + FOb

#
# Compare output to entry
#

# the output layer gives the predicted y (additional non-linearities can be added)
y = FO 
# the L^2 difference between the predicted value and the true one 
mydiff = tf.square(y-y_)

#
# Optimization
#

# the learning rate is a function of gstep 
#learning_rate = tf.train.exponential_decay( 0.05, gstep, 200, 0.97, staircase=True)

learning_rate = tf.train.exponential_decay( learning_rate_start, gstep, learning_rate_global_step, learning_rate_decay_rate, staircase=True)

# the cost function is the mean difference
my_cost = tf.reduce_mean(mydiff)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(my_cost)

# Use ADAM optimization
#train_step_adam = tf.train.AdamOptimizer(10^-3).minimize(my_cost)
train_step_adam = tf.train.AdamOptimizer(learning_rate).minimize(my_cost)


#
# Running the optimization
#

#init = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init) # older versions of TensorFlow
sess.run(tf.global_variables_initializer())
#sess.run(tf.initialize_all_variables())

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_cost = tf.square(y- y_)
accuracy = tf.reduce_mean(tf.cast(pred_cost, tf.float32))

# 
# plotting objects
#
a_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings_a), 1, keep_dims=True))
a_normalized_embeddings = embeddings_a / a_norm
b_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings_b), 1, keep_dims=True))
b_normalized_embeddings = embeddings_b / b_norm

#
#
#
batch_size_act = np.min([batch_size,len(train_x)] );
print('Actual batch size: '+str(batch_size_act))

#saver = tf.train.Saver() # used to save results
#myscore = 1.0 # used for saving
tmppop = range(len(train_x))

final_embeddings_a, final_embeddings_b, final_a_norm, final_b_norm = sess.run([a_normalized_embeddings, b_normalized_embeddings, a_norm, b_norm])
#final_embeddings_a, final_embeddings_b = sess.run([embeddings_a, embeddings_b])
print(final_embeddings_a)
tmpscore = sess.run(accuracy, feed_dict={x: train_x, y_: train_y,  gstep: 0})
print('iter pre'+', current score (training set): '+str(tmpscore))

for i in range(num_iter):
    # Choose batch_size_act elements randomly
    dataperm_ids_tmp = random.sample(tmppop, k=batch_size_act )
    #print(dataperm_ids_tmp)
    #train_step_adam.run( feed_dict={x: train_x[dataperm_ids_tmp,:], y_: train_y[dataperm_ids_tmp], gstep: i })
    #train_step.run( feed_dict={x: train_x[dataperm_ids_tmp,:], y_: train_y[dataperm_ids_tmp], gstep: i })
    # Run a step of the iteration, record the learning rate
    tmprate = learning_rate.eval(feed_dict={ gstep: i })
    if (i<optimization_how_many_adam):
        tmprate, _  = sess.run((learning_rate, train_step_adam), feed_dict={x: train_x[dataperm_ids_tmp,:], y_: train_y[dataperm_ids_tmp], gstep: i })
    else:
        tmprate, _  = sess.run((learning_rate, train_step), feed_dict={x: train_x[dataperm_ids_tmp,:], y_: train_y[dataperm_ids_tmp], gstep: i })
        
    if ((i % 1000) == 0): # Let us know that you are still alive
        print('iter '+str(i)+' - '+myrunstr )
    if ((i % show_scores_once_in) == 0): # Give us some scores once a while
        print('iter '+str(i)+', current learning rate is '+str(tmprate))
        # Accuracy on the entire training set (slow...)
        #tmpscore = sess.run(accuracy, feed_dict={x: train_x, y_: train_y,  gstep: i})
        tmpscore = accuracy.eval( feed_dict={x: train_x, y_: train_y,  gstep: i})
        print('iter '+str(i)+', current score (training set): '+str(tmpscore))
        # Accuracy on the entire test set (slow...)
        tmptestscore = sess.run(accuracy, feed_dict={x: test_x, y_: test_y,  gstep: i})
        print('iter '+str(i)+', current score (test set)    : '+str(tmptestscore))
        #if (tmpscore <= myscore):
        #    #tmppath = saver.save(sess, "test_save")
        #    #print("model saved")
        #    myscore = tmpscore
    if ((i>0)&(((i)%show_results_once_in)==0)):
        # The normalized objects have fewer outliers
        final_embeddings_a, final_embeddings_b, final_a_norm, final_b_norm = sess.run([a_normalized_embeddings, b_normalized_embeddings, a_norm, b_norm])
        #final_embeddings_a, final_embeddings_b, final_a_norm, final_b_norm = sess.run([a_normalized_embeddings, b_normalized_embeddings, a_norm, b_norm])
        # The actual representations:
        #final_embeddings_a, final_embeddings_b = sess.run([embeddings_a, embeddings_b])

        pltcnt=pltcnt+1
        
        #
        # TSNE Embedding
        #
        
        tsne_a = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_a = tsne_a.fit_transform(final_embeddings_a)
        tsne_b = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_b = tsne_b.fit_transform(final_embeddings_b)
        plt.figure(pltcnt)
        
        plt.subplot(321)
        plt.scatter(low_dim_embs_a[:,0],low_dim_embs_a[:,1],c=list(range(La)) ) 
        plt.title('Iteration # '+str(i)+'('+myrunstr+') '+'(dims: '+str(La)+','+str(Lb)+','+str(Ly)+').')
        plt.subplot(322)      
        plt.scatter(low_dim_embs_b[:,0],low_dim_embs_b[:,1],c=list(range(Lb)))
        
        #
        # Diffusion Maps embedding
        #
        
        
        MapEmbd_a, UWeighted_a, myd_a, svals_a, ker2_a, A_a = diffmaps.vecToDiffMap( final_embeddings_a , DM_islocal, numneigh=DM_numneigh,myeps=None,islocal=1,isepsviasquare=0)
        print('Diffusion maps eigenvalues (a)')
        print (svals_a[0:8])
        plt.subplot(323)
        plt.scatter( MapEmbd_a[:,1], MapEmbd_a[:,2], c=list(range(La)) )

        MapEmbd_b, UWeighted_b, myd_b, svals_b, ker2_b, A_b = diffmaps.vecToDiffMap( final_embeddings_b , DM_islocal, numneigh=DM_numneigh,myeps=None,islocal=1,isepsviasquare=0)
        print('Diffusion maps eigenvalues (b)')
        print (svals_b[0:8])
        plt.subplot(324)
        plt.scatter( MapEmbd_b[:,1], MapEmbd_b[:,2], c=list(range(Lb)) )
        
        #
        # Try to get the entire dataset at all points in the matrix
        #
        #tmpy = sess.run(y, feed_dict={x: datx, y_: 0*daty, gstep : 1 })
        tmpy = y.eval( feed_dict={x: datx, y_: 0*daty, gstep : 1 })
        #tmpy = y.eval( feed_dict={x: train_x, y_: 0*train_y, gstep : 1 })
        # reformat into the original form
        dat_re = np.zeros([La,Lb,Ly])
        #print(len(datx))
        for j1 in range(len(datx)):
            dat_re[np.int(datx[j1,0]), np.int(datx[j1,1])] =tmpy[j1]
        #print(len(train_x))
        #print(train_y)
        #print(tmpy)
        #print(np.max(tmpy))
        #print(np.mean(np.power(tmpy-train_y,2)))
        #for j1 in range(len(train_x)):
        #    dat_re[np.int(train_x[j1,0]), np.int(train_x[j1,1])] =tmpy[j1]
        #print(dat_re.shape)
        plt.subplot(325) 
        #print(dat_re.shape)
        plt.imshow(dat_re)
        #plt.imshow(dat_re[:,:,0:3])
        plt.imshow(dat_re[:,:,0:imshow_dims])
        
        
        #
        # A hack to print while running, may fail in different versions
        #
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        



print(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

input("Press Enter to continue...")

