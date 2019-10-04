# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:32:56 2019

@author: jayspc
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot = True)

'''
one_hot does this , if we have multiclass classification
like 0-9
it lits up the requried digit as 1 and the rest are 0's like digital electronics would do
say 
0 = [1,0,0,0,0,0,0,0,0,0],
3 = [0,0,1,0,0,0,0,0,0,0]..
so on and so forth
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')


#weights is a tensorflow variable really nothing but simply 
#weights = random_normal[] in python


def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'baises': tf.variable(tf.random_normal(n_nodes_hl1))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'baises': tf.variable(tf.random_normal(n_nodes_hl2))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'baises': tf.variable(tf.random_normal(n_nodes_hl3))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'baises': tf.variable(tf.random_normal(n_classes))}
    
    
    l1 = tf.add(tf.multiply(data, hidden_1_layer['weights']) + hidden_1_layer['baises'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.multiply(l1, hidden_2_layer['weights']) + hidden_2_layer['baises'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.multiply(l2, hidden_3_layer['weights']) + hidden_3_layer['baises'])
    l3 = tf.nn.relu(l3)
    
    output = tf.multiply(tf.multiply(l3, output_layer['weights']) + output_layer['baises'])
    
    return output # remember output is one-hot array 
    
 
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) #adamOptimizer has a parameter learning rate default set to 0.001
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in hm_epochs:
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
               x,y = mnist.train.next_batch(batch_size)
               _,c = sess.run([optimizer,cost],feed_dict = {x:x ,y: y})
               epoch_loss+=c
            print('Epoch')
                
                








    
    
