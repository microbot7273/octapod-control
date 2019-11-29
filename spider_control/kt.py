#!/usr/bin/python3 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
#import pyplot as plt
import math
import time

class MultiClassLogistic:

    def __init__(self, neurons, layers):
        self.neurons = neurons
        self.layers =  layers
        self.mygraph = tf.Graph()
        self.weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.neuron_input = tf.compat.v1.placeholder(tf.float32, shape=(self.neurons, 1))
        self.weight_in = tf.get_variable(name="Weight_input", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.weight_initer)
        self.neuron_hid = tf.compat.v1.placeholder(tf.float32, shape=(int(self.neurons/2), 1))
        self.weight_initer1 = tf.truncated_normal_initializer(mean=1.0, stddev=0.01)
        self.weight_hid = tf.get_variable(name="Weight_hidden", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.weight_initer1)
        self.neuron_out = tf.compat.v1.placeholder(tf.float32, shape=(4, 1))
        self.weight_initer2 = tf.truncated_normal_initializer(mean=2.0, stddev=0.01)
        self.weight_out = tf.get_variable(name="Weight_output", dtype=tf.float64, shape=[4, 2], initializer=self.weight_initer2)
        self.bias_initer =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_in  =tf.get_variable(name="Bias_input", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.bias_initer)
        self.bias_initer1 =tf.truncated_normal_initializer(mean=0.2, stddev=0.01)
        self.bias_hid = tf.get_variable(name="Bias_hidden", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.bias_initer1)
        self.bias_initer2 =tf.truncated_normal_initializer(mean=0.3, stddev=0.01)
        self.bias_out = tf.get_variable(name="Bias_output", dtype=tf.float64, shape=[4, 1], initializer=self.bias_initer2)
        
    #defining input function that can accept the output from previous layer and can calculate the input for present layer

    #def input_function(self, a = [], b = [], c = []):
    #    'where b=output value of neurons from previous layer, a = weights of neurons from the present layer, c = bias'
    #    multiplication = tf.multiply(a, b)
    #    add  = tf.add(multiplication, c)
    #    sess = tf.Session()
    #    n_input = sess.run(add)
    #    return n_input
    #function that returns the softmax of the output layer or any given array of neurons
    def out_softmax(self, neuron_out = []):
        output = tf.nn.softmax(neuron_out)
        with tf.Session() as sess:
            out = sess.run(output)
        return out

    def kt_learning(self, learning_rate, n_out = [], n_preferred = []):
        'where n_out is the actual output and n_preffered is the prefered output of the network'
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=n_out, logits=n_preferred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        with tf.Session() as sess:
            sess.run(optimizer)
        return

    #def get_neuron(self, a, b):
    #    #'while a being the name of the neuron and b being the value of that neuron'
    #    with tf.Session() as sess_n:
    #        sess_n.run(a, feed_dict={a: b})
    #'a function to findout the input of the neurn giving its weights and biases, can only be used in neurons in hidden and output layer'
    #def input_method(self, a=[], b=[], c=[]):
    #    multiplication = tf.multiply(a, b)
    #    d  = tf.reduce_sum(multiplication, 0)
    #    add = tf.add(d, c)
    #    with tf.Session() as sess:
    #        out = sess.run(add)
    #    return out
    def normalization(self, inputs):
        a_min = min(inputs)
        a_max = max(inputs)
        for i in range(len(inputs)):
            inputs[i] = (inputs[i]-a_min)/(a_max-a_min)

    def run(self, carollis_input):
        self.normalization(carollis_input)
        #'finding the output of the input layer'
        #with tf.Session() as sess1_2:

        knowledge_input = tf.add(tf.multiply(carollis_input, self.weight_in), self.bias_in)
        
            #'calculating the input for the hidden layer'
        knowledge_hidden = tf.add(tf.multiply(knowledge_input, self.weight_in), self.bias_hid)
        #'calculating the output of hidden layer'
        knowledge_hidden_output = 3.14*(tf.add(tf.multiply(knowledge_hidden, self.weight_hid), self.bias_hid))#input function of hidden layer
        knowledge_hidden_out = tf.nn.leaky_relu(knowledge_hidden_output, alpha=0.01, name='leaky_relu')
          
        with tf.Session() as sess1_2:
            knowledge_hidden_out1 = sess1_2.run(knowledge_hidden_out)
        #'calculating the input of output layer'
        tf.reshape(knowledge_hidden_out1, [4, 2])#for quadrant method
        in_out = tf.add(tf.multiply(knowledge_hidden_out1, self.weight_out), self.bias_out)
        with tf.Session() as s:
            s.run(in_out)
        #'finding the softmax output of the neurons'
        softmax_output = np.array(4)
        softmax_output = self.out_softmax(in_out)  # this gives the softmax output and stores it in the newly created array
        return softmax_output
    def learn(self, preferred_out, soft_out):
        self.kt_learning(0.1, soft_out, preferred_out)
    
def one_hot_encoding(model_type):
    a_1 = np.zeros((3))
    if (model_type == 1):
        hot_encoded_matrix = np.insert(a_1, 0, 1, axis=None)#insert 1 in the first coloumn in x axis
    elif(model_type == 2):
        hot_encoded_matrix = np.insert(a_1, 1, 1, axis=None)
    elif(model_type == 3):
        hot_encoded_matrix = np.insert(a_1, 2, 1, axis=None)
    elif(model_type == 4):
        hot_encoded_matrix = np.insert(a_1, 3, 1, axis=None)
    else:
        raise ValueError(f"Value {model_type} is not a valid model number")
    return hot_encoded_matrix

if __name__=='__main__':
    knowledge = MultiClassLogistic(8, 3)
    io = np.array([6.45, 4.54, 7, 8.98, 8.88, 12.34, 25.76, 1.67])
    knowledge_out = knowledge.run(io)