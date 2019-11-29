#!/usr/bin/python3 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
#import pyplot as plt
import math
import time
from sklearn.preprocessing import normalize

class MultiClassLogistic:

    def __init__(self, neurons, layers):
        self.neurons = neurons
        self.layers =  layers
        self.weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.weight_initer1 = tf.truncated_normal_initializer(mean=1.0, stddev=0.01)
        self.weight_initer2 = tf.truncated_normal_initializer(mean=2.0, stddev=0.01)
        self.bias_initer =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_initer1 =tf.truncated_normal_initializer(mean=0.2, stddev=0.01)
        self.bias_initer2 =tf.truncated_normal_initializer(mean=0.3, stddev=0.01)
        
        
    #defining input function that can accept the output from previous layer and can calculate the input for present layer

    def input_function(self, a = [], b = [], c = []):
        'where b=output value of neurons from previous layer, a = weights of neurons from the present layer, c = bias'
        multiplication = tf.multiply(a, b)
        add  = tf.add(multiplication, c)
        sess = tf.Session()
        n_input = sess.run(add)
        return n_input
    #function that returns the softmax of the output layer or any given array of neurons
    def out_softmax(self, neuron_out):
        output = tf.nn.softmax(neuron_out)
        with tf.Session() as sess:
            out = sess.run(output)
        return out

    def kt_learning(self, learning_rate, n_out , n_preferred):
        'where n_out is the actual output and n_preffered is the prefered output of the network'
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=n_out, logits=n_preferred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        with tf.Session() as sess:
            sess.run(optimizer)
        return

    def get_neuron(self, a, b):
        #'while a being the name of the neuron and b being the value of that neuron'
        with tf.Session() as sess_n:
            sess_n.run(a, feed_dict={a: b})
    'a function to findout the input of the neurn giving its weights and biases, can only be used in neurons in hidden and output layer'
    def input_method(self, a=[], b=[], c=[]):
        multiplication = tf.multiply(a, b)
        d  = tf.reduce_sum(multiplication, 0)
        add = tf.add(d, c)
        with tf.Session() as sess:
            out = sess.run(add)
        return out
    def normalization(self, inputs):
        a_min = min(inputs)
        a_max = max(inputs)
        for i in range(len(inputs)):
            inputs[i] = (inputs[i]-a_min)/(a_max-a_min)
    
    def input_weight(self):
        with tf.compat.v1.variable_scope("weight_in", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("weight_input", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.weight_initer)
        return v

    def hid_weight(self):
        with tf.compat.v1.variable_scope("weight_hid", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("weight_hidden", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.weight_initer1)
        return v

    def out_weight(self):
        with tf.compat.v1.variable_scope("weight_out", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("weight_input", dtype=tf.float64, shape=[4, 2], initializer=self.weight_initer2)
        return v

    def bias_in(self):
        with tf.compat.v1.variable_scope("bias_in", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("bias_input", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.bias_initer)
        return v

    def bias_hid(self):
        with tf.compat.v1.variable_scope("bias_hidd", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("bias_hidden", dtype=tf.float64, shape=[self.neurons, 1], initializer=self.bias_initer1)
        return v

    def bias_out(self):
        with tf.compat.v1.variable_scope("bias_outt", reuse=tf.compat.v1.AUTO_REUSE):
            v = tf.compat.v1.get_variable("bias_out", dtype=tf.float64, shape=[4, 1], initializer=self.bias_initer2)
        return v
    def run(self, carollis_input):
        c_in=normalize(carollis_input, axis = 0)
        print('normalized carollis input is \n')
        print(c_in)
        c_in = np.array(c_in)
        c_input = tf.compat.v1.convert_to_tensor(c_in, tf.float64)
        #'finding the output of the input layer'
        weight_i = self.input_weight()
        weight_h = self.hid_weight()
        weight_o = self.out_weight()
        bias_i = self.bias_in()
        bias_h = self.bias_hid()
        bias_o = self.bias_out()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        knowledge_input = tf.add(tf.multiply(c_input, weight_i), bias_i)
        sess.run(knowledge_input)
        knowledge_hidden = tf.nn.leaky_relu(knowledge_input, alpha=0.01) 
        #'calculating the output of hidden layer'
        knowledge_hidden_output = 3.14*(tf.add(tf.multiply(knowledge_hidden, weight_h), bias_h))#input function of hidden layer
        knowledge_hidden_out = tf.nn.leaky_relu(knowledge_hidden_output, alpha=0.01, name='leaky_relu')
          
        sess.run(knowledge_hidden_out)
        #'calculating the input of output layer'
        knowledge_hidden_out = tf.reshape(knowledge_hidden_out, [4, 2])#for quadrant method
        out_mult = tf.multiply(knowledge_hidden_out, weight_o)
        out_add = tf.add(out_mult[:, 0], out_mult[:, 1])
        in_out = tf.add(out_add, bias_o)
        #i_o = sess.run(in_out)
        #r_i_o = np.reshape(i_o, (8, 1))
        #outt = np.add(r_i_o[0:4, 0], r_i_o[4:8, 0])
        #outt = np.reshape(outt, (4, 1))
        #out_multt = tf.placeholder(tf.float64, shape=(4, 1))
        #in_outt = tf.add(out_multt, bias_o)
        output = sess.run(in_out)
        #output = sess.run(in_out)
        #'finding the softmax output of the neurons'
        softmax_output = tf.nn.softmax(in_out)
        output = sess.run(softmax_output)
        return output
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