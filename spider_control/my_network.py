#!/usr/bin/python3 
import tensorflow as tf
import numpy as np
from matplotlib import pylab
import pylab as plt
import math
import time
import leg_dynamics as ld


    
class  Architecture:
    'Common class defining the architecture of the neural network i.e. number of neurons in each layer and number of layers'

    def __init__(self, layers, neurons, number):
        self.layers = layers
        self.neurons = neurons
        self.number = number#number of the model
        self.weight_in = 'weight_in'+str(self.number)
        self.weight_out = 'weight_out'+str(self.number)
        self.weight_hid = 'weight_hid'+str(self.number)
        self.weight_hid_cont = 'weight_hid_cont'+str(self.number)
        self.bias_in = 'bias_in'+str(self.number)
        self.bias_out = 'bias_out'+str(self.number)
        self.bias_hid = 'bias_hid'+str(self.number)
        self.bias_hid_con = 'bias_hid_con'+str(self.number)
        self.neuron_i = tf.compat.v1.placeholder(tf.float32, shape=(neurons, 1))  # _input layer neurons
        self.neuron_o = tf.compat.v1.placeholder(tf.float32, shape=(neurons, 1))  # output layer neurons
        self.neuron_h = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, 1))  # hidden layer neurons, only 8 neurons because there are eight legs
        self.neuron_h_c = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, 1))  # context layer neurons, only 8 neurons because there are eight in hidden layer
        self.weight_initer = tf.truncated_normal_initializer(mean=1.0, stddev=0.01)
        self.weight_initer1 = tf.truncated_normal_initializer(mean=1.0, stddev=0.05)
        self.weight_initer2 = tf.truncated_normal_initializer(mean=1.0, stddev=0.1)
        self.weight_initer3 = tf.truncated_normal_initializer(mean=1.0, stddev=0.15)
        self.weights_i = tf.get_variable(name=self.weight_in ,dtype=tf.float32, shape=[self.neurons, 1], initializer=self.weight_initer)  # weights of input layer neurons
        self.weights_o = tf.get_variable(name=self.weight_out, dtype=tf.float32, shape=[self.neurons, 1], initializer=self.weight_initer1)  # weights of output layer neurons
        self.weights_h = tf.get_variable(name=self.weight_hid, dtype=tf.float32, shape=[self.neurons/3, self.layers], initializer=self.weight_initer2)  # weights of hidden layer neurons
        self.weights_h_c = tf.get_variable(name=self.weight_hid_cont, dtype=tf.float32, shape=[self.neurons/3, self.layers], initializer=self.weight_initer3) # weights of context layer neurons
        self.bias_initer =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_initer1 =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_initer2 =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_initer3 =tf.truncated_normal_initializer(mean=0.1, stddev=0.01)
        self.bias_i = tf.get_variable(name=self.bias_in, dtype=tf.float32, shape=[self.neurons, 1], initializer=self.bias_initer) #biases of each neurons
        self.bias_o = tf.get_variable(name=self.bias_out, dtype=tf.float32, shape=[self.neurons, 1], initializer=self.bias_initer1)
        self.bias_h = tf.get_variable(name=self.bias_hid,  dtype=tf.float32, shape=[self.neurons/3, 1], initializer=self.bias_initer2)
        self.bias_h_c = tf.get_variable(name=self.bias_hid_con, dtype=tf.float32, shape=[self.neurons/3, 1], initializer=self.bias_initer3)
     # get neuron value method helps us to write the joint angles to the individual neurons which we can later use for learning
    def get_neuron_value(self, a, b):
        with tf.Session() as sess_n:
            sess_n.run(a, feed_dict={a: b})
        return
    def input_method(self, neurons = [], weights = [], bias = []):
        #input function of the main neural network
        multi = tf.matmul(neurons, weights, transpose_a = False, transpose_b=False)#use False as tf.False if this does not work
        add1 = tf.add(multi, bias)
        ses = tf.Session()
        neuron_input = ses.run(add1)
        return neuron_input
    def transfer_functions(self, a = []):
        b = tf.nn.softmax(a, name='sigmoid')
        #0.5 being the threshold that is required to activate the neuron
        if (b >= 0.5):
            out = b*a
        else:
            out = 0
        return out
    def NN_learningPunishment(self, LearningRate, punishment):
        #a method to minimize the rewards and alot the weights based on that
        optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(punishment)
        with tf.Session as sess:
            sess.run(optimizer)
        return
    def NN_learningReward(self, LearningRate, reward):
        #a method to maximize reward and alot the weights based on that
        optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize((-1*reward))
        with tf.Session as session:
            session.run(optimizer)
    def output_function(self, a = []):
        b = tf.nn.softmax(a, name='sigmoid')
        out = b*a
        return out
    def nn_run(self, joint_angles = []):
        #function that directly runs the neural network and returns the prediction
        output_layer1 = np.array(24)
        input_layerhid = np.array(8)
        output_layerhid = np.array(8)
        input_layerhidcont = np.array(8)
        output_layerhidcont = np.array(8)
        input_layerout = np.array(24)
        output_layerout = np.array(24)
        self.get_neuron_value(self.neuron_i, joint_angles)
        for i in range(0, 24, 1):
            output_layer1[i] = self.input_method(self.neuron_i[i, :], self.weights_i[:, i])  # finding the ouput of the input layer
        for j in range(0, 8, 1):
            output_layerh = output_layer1[0, 3 * j:3 * j + 3]
            output_layerh.append(output_layerhidcont[j])  # adding output from context layer from previous iteration as input for hidden layer
            input_layerhid[j] = self.input_method(output_layer1, self.weights_h[j, :])
            self.get_neuron_value(self.neuron_h, input_layerhid)  # feeding the neuron value of hidden layer
            output_layerhid = self.transfer_functions(self.neuron_h)
        for i in range(0, 8, 1):
            input_layerhidcont[i] = self.input_method(output_layerhid[i], self.weights_h_c[i])  # input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
            self.get_neuron_value(self.neuron_h_c, input_layerhidcont)  # passing the input to the hidden context
            output_layerhidcont = self.transfer_frunctions(self.neuron_h_c)  # output of the neurons from context layer
            input_layerout = tf.multiply(output_layerhid, self.weights_o[0:8])
            input_layerout.append(tf.multiply(output_layerhid, self.weights_o[8:16]))
            input_layerout.append(tf.multiply(output_layerhid, self.weights_o[16:24]))  # element wise multiplication
            self.get_neuron_value(self.neuron_o, input_layerout)
            output_layerout = self.output_function(self.neuron_o) # output of the neural network
        return output_layerout
    def nn_learn(self,  rew):
        Learningrate = 0.1
        self.NN_learningReward(Learningrate, rew)


    

