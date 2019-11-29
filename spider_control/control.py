#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import roslib
roslib.load_manifest('joint_states_listener')
roslib.load_manifest('spider_control')
from matplotlib import pylab
import pylab as plt
import math
import threading
from geometry_msgs import *
import rospy, yaml, sys
#from osrf_msgs.msg import JointCommands
from sensor_msgs.msg import JointState
from joint_states_listener.srv import *
#from joint_States_listener.srv import ReturnJointStates
import threading
import time
from std_msgs.msg import Float64
from std_msgs.msg import Header
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

#tf.compat.v1.enable_eager_execution()

#global g_joint_states
#global g_position
#global g_pos1
g_joint_states = None
g_position = None #np.ones(24)#np.empty(24, dtype = int)
g_pos1 = None
#global tactile_states
tactile_states = None #np.ones(8)#np.empty(8, dtype = int)
def tactile_callback(data):
    rospy.loginfo(msg.data)
    global tactile_states
    tactile_states = data.data

def joint_callback(data):
    rospy.loginfo(data.position)
    global g_joint_states 
    global g_position
    global g_pos1
    g_joint_states = data
    #for i in len(data.position):
        #g_position[i] = data.position[i]
    g_position = data.position
    if len(data.position) > 0:
        print("jointstate more than 0")
        g_pos1 = data.position[0]
    #print(g_position)

def timer_callback(event): # Type rospy.TimerEvent
    print('timer_cb (' + str(event.current_real) + '): g_positions is')
    print(str(None) if g_positions is None else str(g_positions))

def timer_callback_tactile(event):
    print('timer_cb_t('+str(event.current_real) + '): tactile_states is')
    print(str(None) if tactile_states is None else str(tactile_states))


    

def joint_modifier(*args):
    #choice describes what the node is supposed to do whether act as publisher or subscribe to joint states or tactile sensors
    rospy.init_node('joint_listener_publisher', anonymous=True)
    pub1 = rospy.Publisher('joint_states', JointState, queue_size = 10)
    if(len(args)>1):
        choice = args[0]
        joint_name = args[1]
        position = args[2]
    else:
        choice = args[0]
    if (choice == 1):
        rate = rospy.Rate(1)
        robot_configuration = JointState()
        robot_configuration.header = Header()
        robot_configuration.name = [joint_name]
        robot_configuration.position = [position]
        robot_configuration.velocity = [10]
        robot_configuration.effort = [100]
        while not rospy.is_shutdown():
            robot_configuration.header.stamp = rospy.Time.now()
            rospy.loginfo(robot_configuration)
            break
        pub1.publish(robot_configuration)
        rospy.sleep(2)
    if (choice == 2):
        #rospy.Timer(rospy.Duration(2), joint_modifier)
        rospy.Subscriber("joint_states", JointState, joint_callback)
        #rospy.Timer(rospy.Duration(2), timer_callback)
        rospy.spin()
    if (choice == 3):
        #rospy.Timer(rospy.Duration(2), joint_modifier)
        rospy.Subscriber("/sr_tactile/touch/ff", Float64, tactile_callback)
        #rospy.Timer(rospy.Duration(2), timer_callback_tactile)
        rospy.spin()
 
    

    




class  Leg_attribute:
    m_l = 1  # mass of the link in every leg
    m_b = 4  # mass of the body
    x_b = 0
    y_b = 0
    l_1 = 4  # length of the links 1 and 2 of every leg
    l_3 = 6  # length of the link 3 from every leg i.e. distal joint
    r_1 = 2  # centre of gravity of links 1 and 2 from every leg
    r_3 = 3  # centre of gravity of links 3 from every leg i.e. distal link from distal joint
    radius = 0.5 #radius of the links
    global acc #acceleration at which the robot is supposed to move
    j_angles = [] #np.ones((1, 3)) # in radians
    j_velocities = [] #np.ones((1,3))
    j_efforts = [] #np.ones((1, 3))
    #touch = False #variable to detect whether tactile sensor touch the ground or not
    def __init__(self, position1, position2, position3, velocity1, velocity2, velocity3, effort1, effort2, effort3, acceleration):
        self.j_angles[0] = position1  # vector containing the joint angles
        self.j_angles[1] = position2
        self.j_angles[2] = position3
        self.j_velocities[0] = velocity1  # vector containing joint velocities
        self.j_velocities[1] = velocity2
        self.j_velocities[2] = velocity3
        self.j_efforts[0] = effort1  # vector containing efforts of each joint in the leg
        self.j_efforts[1] = effort2
        self.j_efforts[2] = effort3
        self.acc = acceleration

    def give_angles(self, j_angles):
        a = j_angles
        return a


    def x_left_com(self, a, b, c, d, one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return (1/(2*(a+6*b)))*(a*d+2*b*c(3*(math.cos(one_three)+math.cos(three_three)+math.cos(five_three)+math.cos(seven_three))+2*(math.cos(one_two)+math.cos(three_two)+math.cos(five_two)+math.cos(seven_two))+math.cos(one_one)+math.cos(three_one)+math.cos(five_one)+math.cos(seven_one)))

   #a,b,c,d,e,f,j,k,l,m,n,o,p are respectively m_b,m_l,L,x_body_com/y_body_com,theta_1_1,theta_1_2,...theta_8_3

    def y_left_com(self, a, b, c, d,  one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return (1/(2*(a+6*b)))*(a*d+2*b*c(3*(math.sin(one_three)+3*math.sin(three_three)+math.sin(five_three)+math.sin(seven_three))+2*(math.sin(one_two)+math.sin(three_two)+math.sin(five_two)+math.sin(seven_two))+math.sin(one_one)+math.sin(three_one)+math.sin(five_one)+math.sin(seven_one)))

    def x_right_com(self, a, b, c, d, one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return (1/(2*(a+6*b)))*(3*a*d+2*b*c*(3*math.cos(two_three)+3*math.cos(four_three)+3*math.cos(six_three)+3*math.cos(eight_three)+2*math.cos(two_two)+2*math.cos(four_two)+2*math.cos(six_two)+2*math.cos(eight_two)+math.cos(two_one)+math.cos(four_one)+math.cos(six_one)+math.cos(eight_one)))

    def y_right_com(self, a, b, c, d, one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return (1/(2*(a+6*b)))*(3*a*d+2*b*c*(3*math.sin(two_three)+3*math.sin(four_three)+3*math.sin(six_three)+3*math.sin(eight_three)+2*math.sin(two_two)+2*math.sin(four_two)+2*math.sin(six_two)+2*math.sin(eight_two)+math.sin(two_one)+math.sin(four_one)+math.sin(six_one)+math.sin(eight_one)))

    def x_system_com(self, a, b, c, d, one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return  (1/(a+6*b*c))*(2*a*d+b*c*(3*(math.cos(one_three)+math.cos(two_three)+math.cos(three_three)+math.cos(four_three)+math.cos(five_three)+math.cos(six_three)+math.cos(seven_three)+math.cos(eight_three))+2*(math.cos(one_two)+math.cos(two_two)+math.cos(three_two)+math.cos(four_two)+math.cos(five_two)+math.cos(six_two)+math.cos(seven_two)+math.cos(eight_two))+math.cos(one_one)+math.cos(two_one)+math.cos(three_one)+math.cos(four_one)+math.cos(five_one)+math.cos(six_one)+math.cos(seven_one)+math.cos(eight_one)))

    def y_system_com(self, a, b, c, d, one_one, one_two, one_three, two_one, two_two, two_three, three_one, three_two, three_three, four_one, four_two, four_three, five_one, five_two, five_three, six_one, six_two, six_three, seven_one, seven_two, seven_three, eight_one, eight_two, eight_three):
        return  (1/(a+6*b*c))*(2*a*d+b*c*(3*(math.sin(one_three)+math.sin(two_three)+math.sin(three_three)+math.sin(four_three)+math.sin(five_three)+math.sin(six_three)+math.sin(seven_three)+math.sin(eight_three))+2*(math.sin(one_two)+math.sin(two_two)+math.sin(three_two)+math.sin(four_two)+math.sin(five_two)+math.sin(six_two)+math.sin(seven_two)+math.sin(eight_two))+math.sin(one_one)+math.sin(two_one)+math.sin(three_one)+math.sin(four_one)+math.sin(five_one)+math.sin(six_one)+math.sin(seven_one)+math.sin(eight_one)))

    def mid_point_x(self, x_left, x_right):
        'in order to calculate whether the syste com is normal to the line between left and right com'
        x_mid = (x_right + x_left)/2
        return x_mid

    def mid_point_y(self, y_left, y_right):
        'in order to calculate whether the system com is normal to the line between left and right com'
        y_mid = (y_right + y_left)/2
        return y_mid

    def slope(self, x_mid, y_mid, x_system, y_system):
        'slope to measure the balance of the robot that can be used to define reward or punishment'
        m = (y_system - y_mid)/(x_system - x_mid)
        m_radian = (m*3.14)/180
        return m_radian

    #calculates carolis term for each leg
    def carolis_term(self, acc, m_l, r_1, l_1, j_angles=[], j_velocities=[]):
        term_1 = acc*(m_l*9.8*j_velocities[0]*(r_1*math.sin(j_angles[0])+2*l_1*math.sin(j_angles[0])+r_1*math.sin(j_angles[0]+j_angles[1])+r_1*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))+m_l*9.8*j_velocities[1]*(r_1*math.sin(j_angles[0]+j_angles[1])+l_1*math.sin(j_angles[1])+r_1*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))+m_l*9.8*j_velocities[2]*(r_1*math.sin(j_angles[0]+j_angles[1]+j_angles[2])))
        term_2 = acc*(m_l*9.8*j_velocities[0]*(r_1*math.sin(j_angles[0]+j_angles[1])+r_1*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))+m_l*9.8*j_velocities[1]*(r_1*math.sin(j_angles[0]+j_angles[1])+l_1*math.sin(j_angles[1])+r_1*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))+m_l*9.8*j_velocities[2]*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))
        term_3 = acc*(m_l*9.8*j_velocities[0]*math.sin(j_angles[0]+j_angles[1]+j_angles[2])+m_l*9.8*j_velocities[1]*math.sin(j_angles[0]+j_angles[1]+j_angles[2])+m_l*9.8*j_velocities[2]*math.sin(j_angles[0]+j_angles[1]+j_angles[2]))
        term = term_1 +term_2 + term_3
        return term

















class  Architecture:
    'Common class defining the architecture of the neural network i.e. number of neurons in each layer and number of layers'

    def __init__(self, layers, neurons):
        self.layers = layers
        self.neurons = neurons
        #global neuron_i
        #global neuron_o
        #global neuron_h
        #global neuron_h_c
        #global weights_i
        #global weights_o
        #global weights_h
        #global weights_h_c
        #global bias_i
        #global bias_o
        #global bias_h
        #global bias_h_c
        self.neuron_i = tf.compat.v1.placeholder(tf.float32, shape=(neurons, 1))  # _input layer neurons
        self.neuron_o = tf.compat.v1.placeholder(tf.float32, shape=(neurons, 1))  # output layer neurons
        self.neuron_h = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, 1))  # hidden layer neurons, only 8 neurons because there are eight legs
        self.neuron_h_c = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, 1))  # context layer neurons, only 8 neurons because there are eight in hidden layer
        self.weights_i = tf.compat.v1.placeholder(tf.float32, shape=(neurons, layers))  # weights of input layer neurons
        self.weights_o = tf.compat.v1.placeholder(tf.float32, shape=(neurons, layers))  # weights of output layer neurons
        self.weights_h = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, layers))  # weights of hidden layer neurons
        self.weights_h_c = tf.compat.v1.placeholder(tf.float32, shape=(neurons/3, layers))  # weights of context layer neurons
        self.bias_i = tf.random_uniform((neurons, 1), minval = 0, maxval = 1, dtype = tf.float32, seed=9999) #biases of each neurons
        self.bias_o = tf.random_uniform((neurons, 1), minval = 0, maxval = 1, dtype = tf.float32, seed = 1111)
        self.bias_h = tf.random_uniform((int(neurons/3), 1), minval = 0, maxval = 1, dtype = tf.float32, seed=2222)
        self.bias_h_c = tf.random_uniform((8, 1), minval = 0, maxval = 1, dtype = tf.float32, seed=3333)

    def weight_initialize (self, neurons):
        weights1 = tf.Variable(tf.random_uniform((neurons, 1), minval=0, maxval=1, dtype=tf.float32, seed=7273))
        weights2 = tf.Variable(tf.random_uniform((neurons, 1), minval=0, maxval=1, dtype=tf.float32, seed=3000))
        weights3h=tf.Variable(tf.random_uniform((neurons/3, neurons/6), minval=0, maxval=1, dtype=tf.float32, seed= 119))
        weights3c=tf.Variable(tf.random_uniform((neurons/3, 1), minval=0, maxval=1, dtype=tf.float32, seed=1508))
        sess=tf.Session()
        weights_i, weights_o, weights_h, weights_h_c = sess.run(weights1, weights2, weights3h, weights3c)
        return weights_i, weights_o, weights_h, weights_h_c

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

    def transfer_function(self, a = []):
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





 #class that describes multi class logistic regression for knowledge transfer
class MultiClassLogistic:

    def __init__(self, neurons, layers):
        self.neurons = neurons
        self.layers =  layers
        global weight_in, weight_hid, weight_out, bias_hid, bias_in, bias_out
        #self.weight_in = weight_in
        #self.weight_hid = weight_hid
        #self.weight_out = weight_out
        #self.bias_hid = bias_hid
        #self.bias_out = bias_out
        #self.bias_in = bias_in
        #self.neuron_input = tf.compat.v1.placeholder(tf.float32, shape=(neurons, 1))
        #self.neuron_hidden = tf.compat.v1.placeholder(tf.float32, shape=(neurons/4, 1))
        #self.neuron_out = tf.compat.v1.placeholder(tf.float32, shape=(4, 1))#four neurons in output layer because of the four quadrants
        weight_in = tf.get_variable("weight_in", [neurons, 1])
        weight_hid = tf.get_variable("weight_hid", [int(neurons/2), int(neurons/2)])
        weight_out = tf.get_variable("weight_out", [4, 2])
        weights1 = np.random.rand(neurons, 1)#weights of input layer
        weights2 = np.random.rand(int(neurons/2), int(neurons/2))#weights for hidden layer, number of neurons in hidden layer is identified by the quadrant concept
        weights3 = np.random.rand(4, 2)#weights for output layer
        bias_in  =tf.random_normal((neurons, 1), dtype = tf.float32, seed = 2009)
        bias_hid = tf.random_normal((int(neurons/2), 1), dtype = tf.float32, seed = 2509)
        bias_out = tf.random_normal((4, 1), dtype=tf.float32, seed = 1234)
        with tf.Session() as session1:
            session1.run(weight_in, feed_dict={weight_in:weights1})
        with tf.Session() as session2:
            session2.run(weight_hid, feed_dict={weight_hid:weights2})
        with tf.Session() as session3:
            session3.run(weight_out, feed_dict={weight_out:weights3})
        with tf.Session() as session4:
            bias_in, bias_hid, bias_out  =session4.run([bias_in, bias_hid, bias_out])

    #defining input function that can accept the output from previous layer and can calculate the input for present layer

    def input_function(self, a = [], b = [], c = []):
        'where b=output value of neurons from previous layer, a = weights of neurons from the present layer, c = bias'
        multiplication = tf.multiply(a, b)
        add  = tf.add(multiplication, c)
        sess = tf.Session()
        n_input = sess.run(add)
        return n_input
    #function that returns the softmax of the output layer or any given array of neurons
    def out_softmax(self, neuron_out = []):
        output = tf.nn.softmax(neuron_out)
        with tf.Session() as sess:
            out = sess.run(output)
        return out

    #def output_func(self, ):

    def activation(self, a):
        'where a is the input of the neuron and the activation function is modelled as rectified linear output'
        if(a>=0):
            b = a
        else:
            b = 0
        return b
    def kt_learning(self, learning_rate, n_out = [], n_preferred = []):
        'where n_out is the actual output and n_preffered is the prefered output of the network'
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=n_out, logits=n_preferred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        with tf.Session() as sess:
            sess.run(optimizer)
        return

    def get_neuron(self, a, b):
        'while a being the name of the neuron and b being the value of that neuron'
        with tf.Session() as sess_n:
            sess_n.run(a, feed_dict={a: b})
        return
    'a function to findout the input of the neurn giving its weights and biases, can only be used in neurons in hidden and output layer'
    def input_method(self, a=[], b=[], c=[]):
        multiplication = tf.multiply(a, b)
        d  = tf.reduce_sum(multiplication, 0)
        add = tf.add(d, c)
        with tf.Session() as sess:
            out = sess.run(add)
        return out










    #function to normalize the input data
def normalization(a):
    a_min = min(a)
    a_max = max(a)
    for i in range(len(a)):
        a[i] = (a[i]-a_min)/(a_max-a_min)
    return

def one_hot_encoding( iteration_time, model_type):
    a_1 = np.zeros((3, 1))
    if (model_type == 1):
        
        hot_encoded_matrix = np.insert(a_1, 0, 1, axis=0)#insert 1 in the first coloumn in x axis
    elif(model_type == 2):
        
        hot_encoded_matrix = np.insert(a_1, 1, 1, axis=0)
    elif(model_type == 3):
        
        hot_encoded_matrix = np.insert(a_1, 2, 1, axis=0)
    elif(model_type == 4):
       
        hot_encoded_matrix = np.insert(a_1, 3, 1, axis=0)
    else:
        raise ValueError(f"Value {model_type} is not a valid model number")
    return hot_encoded_matrix


if __name__=="__main__":
    spiderJoint1Names = ['joint_1_1', 'joint_1_2', 'joint_1_3']
    spiderJoint2Names = ['joint_2_1', 'joint_2_2', 'joint_2_3']
    spiderJoint3Names = ['joint_3_1', 'joint_3_2', 'joint_3_3']
    spiderJoint4Names = ['joint_4_1', 'joint_4_2', 'joint_4_3']
    spiderJoint5Names = ['joint_5_1', 'joint_5_2', 'joint_5_3']
    spiderJoint6Names = ['joint_6_1', 'joint_6_2', 'joint_6_3']
    spiderJoint7Names = ['joint_7_1', 'joint_7_2', 'joint_7_3']
    spiderJoint8Names = ['joint_8_1', 'joint_8_2', 'joint_8_3']
    with tf.Session() as se1:
        se1.run(tf.global_variables_initializer())
    model_1 = Architecture(3, 24) #for model one of the terrain
    model_2 = Architecture(3, 24) #for model two of the terrain
    model_3 = Architecture(3, 24) #for model three of the terrain
    model_4 = Architecture(3, 24) #for model four of the terrain
    run_method = input("How do you want to run the algorithm ? for training, type train. for testing, type test")
    knowledge_transfer = MultiClassLogistic(8, 3)
    g = 9.8
    reward_history = np.array([0])
    prev_reward = 0
    m = Leg_attribute.m_l
    r = Leg_attribute.r_1
    l = Leg_attribute.l_1
    global run_time
    global model_trained
    run_time_time = input("please enter how long you intend to run the robot")
    #'giving initial angles '
    initial_angles = np.random.uniform(low = 0, high =2.5, size=24)
    number = 0
    length1 = len(spiderJoint1Names)
    length2 = len(spiderJoint2Names)
    length3 = len(spiderJoint3Names)
    length4 = len(spiderJoint4Names)
    length5 = len(spiderJoint5Names)
    length6 = len(spiderJoint6Names)
    length7 = len(spiderJoint7Names)
    length8 = len(spiderJoint8Names)
    for i1 in range(length1):
        jointname = spiderJoint1Names[i1]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
        print(number)
    for i2 in range(length2):
        jointname = spiderJoint2Names[i2]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i3 in range(length3):
        jointname = spiderJoint3Names[i3]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i4 in range(length4):
        jointname = spiderJoint4Names[i4]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i5 in range(length5):
        jointname = spiderJoint5Names[i5]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i6 in range(length6):
        jointname = spiderJoint6Names[i6]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i7 in range(length7):
        jointname = spiderJoint7Names[i7]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    for i8 in range(length8):
        jointname = spiderJoint8Names[i8]
        joint_modifier(1, jointname, initial_angles[number])
        number+=1
    print("succefully published initial angles")
    joint_modifier(3)#just checking whether tactile callback is working or not
    print("printing tactile states")
    print(tactile_states)#debugging tactile callback
    print("printed tactile states")
    if ( run_method == 'train'):
        training_time = input("please enter how long you intend to train the algorithm")
        run_time = training_time
        model_trained = int(input("which model do you want to train ?, please enter model_number for straight terrain select 1, for accute angled terrain select 2, for obtuse angled terrain select 3, for up/down terrain select 4"))
        output = one_hot_encoding(run_time, model_trained)
        pref_out = tf.placeholder(tf.float32, shape=(4, 1))
        with tf.Session() as sess1:
            sess1.run(pref_out, feed_dict={pref_out: output})
        print("bug finding")
        i=0
        position = np.array(3)
        velocity = np.array(3)
        effort = np.array(3)
        while(run_time):
            joint_modifier(2)
            print("printing g_position")
            print(g_position)#to check the format of g_position
            print("printed g _position")
            leg_1 = Leg_attribute(g_position[0], g_position[1], g_position[2], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_2 = Leg_attribute(g_position[3], g_position[4], g_position[5], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_3 = Leg_attribute(g_position[6], g_position[7], g_position[8], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_4 = Leg_attribute(g_position[9], g_position[10], g_position[11], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_5 = Leg_attribute(g_position[12], g_position[13], g_position[14], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_6 = Leg_attribute(g_position[15], g_position[16], g_position[17], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_7 = Leg_attribute(g_position[18], g_position[19], g_position[20], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            leg_8 = Leg_attribute(g_position[21], g_position[22], g_position[23], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
            (tactile_1, tactile_2, tactile_3, tactile_4, tactie_5, tactile_6, tactile_7, tactile_8) = joint_modifier(3)
            print("not_touch")
            not_touch = 0
            balance_punishment = 0
            if(not_touch<3):
                balance_punishment = 5
            joint_angle_leg_1 = leg_1.give_angles()
            joint_angle_leg_2 = leg_2.give_angles()
            joint_angle_leg_3 = leg_3.give_angles()
            joint_angle_leg_4 = leg_4.give_angles()
            joint_angle_leg_5 = leg_5.give_angles()
            joint_angle_leg_6 = leg_6.give_angles()
            joint_angle_leg_7 = leg_7.give_angles()
            joint_angle_leg_8 = leg_8.give_angles()
            joint_angles_input = np.concatenate(joint_angle_leg_1, joint_angle_leg_2, joint_angle_leg_3, joint_angle_leg_4, joint_angle_leg_5, joint_angle_leg_6, joint_angle_leg_7, joint_angle_leg_8)
            joint_angle_input = joint_angle_input.reshape(24, 1)  # combined all the input angles into a single coloumn vector
            output_layer1 = np.array(24)
            input_layerhid = np.array(8)
            output_layerhid = np.array(8)
            input_layerhidcont = np.array(8)
            output_layerhidcont = np.array(8)
            input_layerout = np.array(24)
            output_layerout = np.array(24)
            if (model_trained == 1):
                model_1.get_neuron_value(model_1.neuron_i, joint_angle_input)
                for i in range(0, 24, 1):
                    output_layer1[i] = model_1.input_method(model_1.neuron_i[i, :], model_1.weight_in[:, i])  # finding the ouput of the input layer
                for j in range(0, 8, 1):
                    output_layerh = output_layer1[0, 3 * j:3 * j + 3]
                    output_layerh.append(output_layerhidcont[j])  # adding output from context layer from previous iteration as input for hidden layer
                    input_layerhid[j] = model_1.input_method(output_layer1, model_1.weights_i[j, :])
                model_1.get_neuron_value(model_1.neuron_h, input_layerhid)  # feeding the neuron value of hidden layer
                output_layerhid = model_1.transfer_function(model_1.neuron_h)
                for i in range(0, 8, 1):
                    input_layerhidcont[i] = model_1.input_method(output_layerhid[i], model_1.weights_h_c[i])  # input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
                model_1.get_neuron_value(model_1.neuron_h_c, input_layerhidcont)  # passing the input to the hidden context
                output_layerhidcont = model_1.transfer_frunction(model_1.neuron_h_c)  # output of the neurons from context layer
                input_layerout = tf.multiply(output_layerhid, model_1.weights_o[0:8])
                input_layerout.append(tf.multiply(output_layerhid, model_1.weights_o[8:16]))
                input_layerout.append(tf.multiply(output_layerhid, model_1.weights_o[16:24]))  # element wise multiplication
                model_1.get_neuron_value(model_1.neuron_o, input_layerout)
                output_layerout = model_1.output_function(model_1.neuron_o) # output of the neural network for model 1 that is to
                try:
                    joint_modifier(1, 'joint_1_1', output_layerout[0])
                    joint_modifier(1, 'joint_1_2', output_layerout[1])
                    joint_modifier(1, 'joint_1_3', output_layerout[2])
                    joint_modifier(1, 'joint_4_1', output_layerout[9])
                    joint_modifier(1, 'joint_4_2', output_layerout[10])
                    joint_modifier(1, 'joint_4_3', output_layerout[11])
                    joint_modifier(1, 'joint_5_1', output_layerout[12])
                    joint_modifier(1, 'joint_5_2', output_layerout[13])
                    joint_modifier(1, 'joint_5_3', output_layerout[14])
                    joint_modifier(1, 'joint_8_1', output_layerout[21])
                    joint_modifier(1, 'joint_8_2', output_layerout[22])
                    joint_modifier(1, 'joint_8_3', output_layerout[23])
                    joint_modifier(1, 'joint_2_1', output_layerout[3])
                    joint_modifier(1, 'joint_2_2', output_layerout[4])
                    joint_modifier(1, 'joint_2_3', output_layerout[5])
                    joint_modifier(1, 'joint_3_1', output_layerout[6])
                    joint_modifier(1, 'joint_3_2', output_layerout[7])
                    joint_modifier(1, 'joint_3_3', output_layerout[8])
                    joint_modifier(1, 'joint_6_1', output_layerout[15])
                    joint_modifier(1, 'joint_6_2', output_layerout[16])
                    joint_modifier(1, 'joint_6_3', output_layerout[17])
                    joint_modifier(1, 'joint_7_1', output_layerout[18])
                    joint_modifier(1, 'joint_7_2', output_layerout[19])
                    joint_modifier(1, 'joint_7_3', output_layerout[20])
                except rospy.ROSInterruptException:
                    pass
                rospy.spin()
            elif(model_trained == 2):
                model_2.get_neuron_value(model_2.neuron_i, joint_angle_input)
                for i in range(0, 24, 1):
                    output_layer1[i] = model_2.input_method(model_2.neuron_i[i, :], model_2.weight_in[:, i])  # finding the ouput of the input layer
                for j in range(0, 8, 1):
                    output_layerh = output_layer1[0, 3 * j:3 * j + 3]
                    output_layerh.append(output_layerhidcont[j])  # adding output from context layer from previous iteration as input for hidden layer
                    input_layerhid[j] = model_2.input_method(output_layer1, model_2.weights_i[j, :])
                model_2.get_neuron_value(model_2.neuron_h, input_layerhid)  # feeding the neuron value of hidden layer
                output_layerhid = model_2.transfer_function(model_2.neuron_h)
                for i in range(0, 8, 1):
                    input_layerhidcont[i] = model_2.input_method(output_layerhid[i], model_2.weights_h_c[i])  # input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
                model_2.get_neuron_value(model_2.neuron_h_c, input_layerhidcont)  # passing the input to the hidden context
                output_layerhidcont = model_2.transfer_frunction(model_2.neuron_h_c)  # output of the neurons from context layer
                input_layerout = tf.multiply(output_layerhid, model_2.weights_o[0:8])
                input_layerout.append(tf.multiply(output_layerhid, model_2.weights_o[8:16]))
                input_layerout.append(tf.multiply(output_layerhid, model_2.weights_o[16:24]))  # element wise multiplication
                model_2.get_neuron_value(model_2.neuron_o, input_layerout)
                output_layerout = model_2.output_function(model_2.neuron_o)  # output of the neural network for model 1
                try:
                    joint_modifier(1, 'joint_7_1', output_layerout[18])
                    joint_modifier(1, 'joint_7_2', output_layerout[19])
                    joint_modifier(1, 'joint_7_3', output_layerout[20])
                    joint_modifier(1, 'joint_2_1', output_layerout[3])
                    joint_modifier(1, 'joint_2_2', output_layerout[4])
                    joint_modifier(1, 'joint_2_3', output_layerout[5])
                    joint_modifier(1, 'joint_1_1', output_layerout[0])
                    joint_modifier(1, 'joint_1_2', output_layerout[1])
                    joint_modifier(1, 'joint_1_3', output_layerout[2])
                    joint_modifier(1, 'joint_4_1', output_layerout[9])
                    joint_modifier(1, 'joint_4_2', output_layerout[10])
                    joint_modifier(1, 'joint_4_3', output_layerout[11])
                    joint_modifier(1, 'joint_3_1', output_layerout[6])
                    joint_modifier(1, 'joint_3_2', output_layerout[7])
                    joint_modifier(1, 'joint_3_3', output_layerout[8])
                    joint_modifier(1, 'joint_6_1', output_layerout[15])
                    joint_modifier(1, 'joint_6_2', output_layerout[16])
                    joint_modifier(1, 'joint_6_3', output_layerout[17])
                    joint_modifier(1, 'joint_5_1', output_layerout[12])
                    joint_modifier(1, 'joint_5_2', output_layerout[13])
                    joint_modifier(1, 'joint_5_3', output_layerout[14])
                    joint_modifier(1, 'joint_8_1', output_layerout[21])
                    joint_modifier(1, 'joint_8_2', output_layerout[22])
                    joint_modifier(1, 'joint_8_3', output_layerout[23])
                except rospy.ROSInterruptException:
                    pass
                rospy.spin()
            elif(model_trained == 3):
                model_3.get_neuron_value(model_3.neuron_i, joint_angle_input)
                for i in range(0, 24, 1):
                    output_layer1[i] = model_3.input_method(model_3.neuron_i[i, :], model_3.weight_in[:, i])  # finding the ouput of the input layer
                for j in range(0, 8, 1):
                    output_layerh = output_layer1[0, 3 * j:3 * j + 3]
                    output_layerh.append(output_layerhidcont[j])  # adding output from context layer from previous iteration as input for hidden layer
                    input_layerhid[j] = model_3.input_method(output_layer1, model_3.weights_i[j, :])
                model_3.get_neuron_value(model_3.neuron_h, input_layerhid)  # feeding the neuron value of hidden layer
                output_layerhid = model_3.transfer_function(model_3.neuron_h)
                for i in range(0, 8, 1):
                    input_layerhidcont[i] = model_3.input_method(output_layerhid[i], model_3.weights_h_c[i])  # input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
                model_3.get_neuron_value(model_3.neuron_h_c, input_layerhidcont)  # passing the input to the hidden context
                output_layerhidcont = model_3.transfer_frunction(model_3.neuron_h_c)  # output of the neurons from context layer
                input_layerout = tf.multiply(output_layerhid, model_3.weights_o[0:8])
                input_layerout.append(tf.multiply(output_layerhid, model_3.weights_o[8:16]))
                input_layerout.append(tf.multiply(output_layerhid, model_3.weights_o[16:24]))  # element wise multiplication
                model_3.get_neuron_value(model_3.neuron_o, input_layerout)
                output_layerout = model_3.output_function(model_3.neuron_o)  # output of the neural network for model 1
                try:
                    joint_modifier(1, 'joint_1_1', output_layerout[0])
                    joint_modifier(1, 'joint_1_2', output_layerout[1])
                    joint_modifier(1, 'joint_1_3', output_layerout[2])
                    joint_modifier(1, 'joint_8_1', output_layerout[21])
                    joint_modifier(1, 'joint_8_2', output_layerout[22])
                    joint_modifier(1, 'joint_8_3', output_layerout[23])
                    joint_modifier(1, 'joint_3_1', output_layerout[6])
                    joint_modifier(1, 'joint_3_2', output_layerout[7])
                    joint_modifier(1, 'joint_3_3', output_layerout[8])
                    joint_modifier(1, 'joint_2_1', output_layerout[3])
                    joint_modifier(1, 'joint_2_2', output_layerout[4])
                    joint_modifier(1, 'joint_2_3', output_layerout[5])
                    joint_modifier(1, 'joint_5_1', output_layerout[12])
                    joint_modifier(1, 'joint_5_2', output_layerout[13])
                    joint_modifier(1, 'joint_5_3', output_layerout[14])
                    joint_modifier(1, 'joint_4_1', output_layerout[9])
                    joint_modifier(1, 'joint_4_2', output_layerout[10])
                    joint_modifier(1, 'joint_4_3', output_layerout[11])
                    joint_modifier(1, 'joint_7_1', output_layerout[18])
                    joint_modifier(1, 'joint_7_2', output_layerout[19])
                    joint_modifier(1, 'joint_7_3', output_layerout[20])
                    joint_modifier(1, 'joint_6_1', output_layerout[15])
                    joint_modifier(1, 'joint_6_2', output_layerout[16])
                    joint_modifier(1, 'joint_6_3', output_layerout[17])
                except rospy.ROSInterruptException:
                    pass
                rospy.spin()
            elif(model_trained == 4):
                model_4.get_neuron_value(model_4.neuron_i, joint_angle_input)
                for i in range(0, 24, 1):
                    output_layer1[i] = model_4.input_method(model_4.neuron_i[i, :], model_4.weight_in[:, i])  # finding the ouput of the input layer
                for j in range(0, 8, 1):
                    output_layerh = output_layer1[0, 3 * j:3 * j + 3]
                    output_layerh.append(output_layerhidcont[j])  # adding output from context layer from previous iteration as input for hidden layer
                    input_layerhid[j] = model_4.input_method(output_layer1, model_4.weights_i[j, :])
                model_4.get_neuron_value(model_4.neuron_h, input_layerhid)  # feeding the neuron value of hidden layer
                output_layerhid = model_4.transfer_function(model_4.neuron_h)
                for i in range(0, 8, 1):
                    input_layerhidcont[i] = model_4.input_method(output_layerhid[i], model_4.weights_h_c[i])  # input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
                model_4.get_neuron_value(model_4.neuron_h_c, input_layerhidcont)  # passing the input to the hidden context
                output_layerhidcont = model_4.transfer_frunction(model_4.neuron_h_c)  # output of the neurons from context layer
                input_layerout = tf.multiply(output_layerhid, model_4.weights_o[0:8])
                input_layerout.append(tf.multiply(output_layerhid, model_4.weights_o[8:16]))
                input_layerout.append(tf.multiply(output_layerhid, model_4.weights_o[16:24]))  # element wise multiplication
                model_4.get_neuron_value(model_4.neuron_o, input_layerout)
                output_layerout = model_4.output_function(model_4.neuron_o)  # output of the neural network for model 1
                try:
                    joint_modifier(1, 'joint_1_1', output_layerout[0])
                    joint_modifier(1, 'joint_1_2', output_layerout[1])
                    joint_modifier(1, 'joint_1_3', output_layerout[2])
                    joint_modifier(1, 'joint_3_1', output_layerout[6])
                    joint_modifier(1, 'joint_3_2', output_layerout[7])
                    joint_modifier(1, 'joint_3_3', output_layerout[8])
                    joint_modifier(1, 'joint_5_1', output_layerout[12])
                    joint_modifier(1, 'joint_5_2', output_layerout[13])
                    joint_modifier(1, 'joint_5_3', output_layerout[14])
                    joint_modifier(1, 'joint_7_1', output_layerout[18])
                    joint_modifier(1, 'joint_7_2', output_layerout[19])
                    joint_modifier(1, 'joint_7_3', output_layerout[20])
                    joint_modifier(1, 'joint_2_1', output_layerout[3])
                    joint_modifier(1, 'joint_2_2', output_layerout[4])
                    joint_modifier(1, 'joint_2_3', output_layerout[5])
                    joint_modifier(1, 'joint_4_1', output_layerout[9])
                    joint_modifier(1, 'joint_4_2', output_layerout[10])
                    joint_modifier(1, 'joint_4_3', output_layerout[11])
                    joint_modifier(1, 'joint_6_1', output_layerout[15])
                    joint_modifier(1, 'joint_6_2', output_layerout[16])
                    joint_modifier(1, 'joint_6_3', output_layerout[17])
                    joint_modifier(1, 'joint_8_1', output_layerout[21])
                    joint_modifier(1, 'joint_8_2', output_layerout[22])
                    joint_modifier(1, 'joint_8_3', output_layerout[23])
                except rospy.ROSInterruptException:
                    pass
                rospy.spin()
            x_left_com = Leg_attribute.x_left_com(m, g, r, l, joint_angle_leg_1[0],
                                                  joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                  joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                  joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                  joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                  joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                  joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                  joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                  joint_angle_leg_8[1], joint_angle_leg_8[2])
            y_left_com = Leg_attribute.y_left_com(m, g, r, l, joint_angle_leg_1[0],
                                                  joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                  joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                  joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                  joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                  joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                  joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                  joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                  joint_angle_leg_8[1], joint_angle_leg_8[2])
            x_right_com = Leg_attribute.x_right_com(m, g, r, l, joint_angle_leg_1[0],
                                                    joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                    joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                    joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                    joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                    joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                    joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                    joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                    joint_angle_leg_8[1], joint_angle_leg_8[2])
            y_right_com = Leg_attribute.y_right_com(m, g, r, l, joint_angle_leg_1[0],
                                                    joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                    joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                    joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                    joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                    joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                    joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                    joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                    joint_angle_leg_8[1], joint_angle_leg_8[2])
            x_system_com = Leg_attribute.x_system_com(m, g, r, l, joint_angle_leg_1[0],
                                                      joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                      joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                      joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                      joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                      joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                      joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                      joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                      joint_angle_leg_8[1], joint_angle_leg_8[2])
            y_system_com = Leg_attribute.y_system_com(m, g, r, l, joint_angle_leg_1[0],
                                                      joint_angle_leg_1[1], joint_angle_leg_1[2], joint_angle_leg_2[0],
                                                      joint_angle_leg_2[1], joint_angle_leg_2[2], joint_angle_leg_3[0],
                                                      joint_angle_leg_3[1], joint_angle_leg_3[2], joint_angle_leg_4[0],
                                                      joint_angle_leg_4[1], joint_angle_leg_4[2], joint_angle_leg_5[0],
                                                      joint_angle_leg_5[1], joint_angle_leg_5[2], joint_angle_leg_6[0],
                                                      joint_angle_leg_6[1], joint_angle_leg_6[2], joint_angle_leg_7[0],
                                                      joint_angle_leg_7[1], joint_angle_leg_7[2], joint_angle_leg_8[0],
                                                      joint_angle_leg_8[1], joint_angle_leg_8[2])
            x_mid_com = Leg_attribute.mid_point_x(x_left_com, x_right_com)
            y_mid_com = Leg_attribute.mid_point_y(y_left_com, y_right_com)
            reward = prev_reward+ Leg_attribute.slope(x_mid_com, y_mid_com, x_system_com, y_system_com)
            Learningrate = 0.1
            model_1.NN_learningReward(Learningrate, reward)
            prev_reward = reward-balance_punishment#considering the punishment if robot happens to fall down
            leg1_carollis = leg_1.carolis_term(m, r, l, joint_angle_leg_1)
            leg2_carollis = leg_2.carolis_term(m, r, l, joint_angle_leg_2)
            leg3_carollis = leg_3.carolis_term(m, r, l, joint_angle_leg_3)
            leg4_carollis = leg_4.carolis_term(m, r, l, joint_angle_leg_4)
            leg5_carollis = leg_5.carolis_term(m, r, l, joint_angle_leg_5)
            leg6_carollis = leg_6.carolis_term(m, r, l, joint_angle_leg_6)
            leg7_carollis = leg_7.carolis_term(m, r, l, joint_angle_leg_7)
            leg8_carollis = leg_8.carolis_term(m, r, l, joint_angle_leg_8)
            carollis_input = np.array([leg1_carollis, leg2_carollis, leg3_carollis, leg4_carollis, leg5_carollis,
                                      leg6_carollis, leg7_carollis, leg8_carollis])
            normalization(carollis_input)
            knowledge_transfer.get_neuron(knowledge_transfer.neuron_input, carollis_input)
            'finding the output of the input layer'
            knowledge_input = knowledge_transfer.input_function(knowledge_transfer.neuron_input,
                                                                knowledge_transfer.weight_in,
                                                                knowledge_transfer.bias_in)
            'calculating the input for the hidden layer'
            tf.reshape(knowledge_input, [4, 2])
            knowledge_hidden = knowledge_transfer.input_method(knowledge_input, knowledge_transfer.weight_hid,
                                                               knowledge_transfer.bias_hid)
            'feeding in the input value of hidden neurons'
            knowledge_transfer.get_neuron(knowledge_transfer.neuron_hidden, knowledge_hidden)
            'calculating the output of hidden layer'
            knowledge_hidden_out = knowledge_transfer.activation(knowledge_transfer.neuron_hidden)
            'calculating the input of output layer'
            np.reshape(knowledge_hidden, [2, 2])
            extra = np.array(knowledge_hidden(1, 3), knowledge_hidden(2, 4))
            knowledge_out = tf.concat([knowledge_hidden, extra], axis=0)
            in_out = knowledge_transfer.input_method(knowledge_out, knowledge_transfer.weight_out, knowledge_transfer.bias_out)
            with tf.Session as s:
                s.run(in_out)
            'feeding the input value of output neurons'
            knowledge_transfer.get_neuron(knowledge_transfer.neuron_out, in_out)
            'finding the softmax output of the neurons'
            softmax_output = np.array(4)
            softmax_output = knowledge_transfer.out_softmax(knowledge_transfer.neuron_out)  # this gives the softmax output and stores it in the newly created array
            MultiClassLogistic.kt_learning(0.1, softmax_output, pref_out)


    elif ( run_method == 'test'):
        run_time = 1
    rospy.spin()
    joint_angles_input = np.array(24)
    output_layerhidcont = np.zeros((8, 1))
    joint_modifier(2)
    while run_time:

        leg_1 = Leg_attribute(g_position[0], g_position[1], g_position[2], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_2 = Leg_attribute(g_position[3], g_position[4], g_position[5], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_3 = Leg_attribute(g_position[6], g_position[7], g_position[8], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_4 = Leg_attribute(g_position[9], g_position[10], g_position[11], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_5 = Leg_attribute(g_position[12], g_position[13], g_position[14], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_6 = Leg_attribute(g_position[15], g_position[16], g_position[17], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_7 = Leg_attribute(g_position[18], g_position[19], g_position[20], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)
        leg_8 = Leg_attribute(g_position[21], g_position[22], g_position[23], velocity1 = 10, velocity2 = 10, velocity3 = 10, effort1 = 100, effort2 = 100, effort3 = 100, acceleration=1)

        (tactile_1, tactile_2, tactile_3, tactile_4, tactile_5, tactile_6, tactile_7, tactile_8) = joint_modifier(3)
        not_touch = 0
        balance_punishment = 0
        if (tactile_1 == 0.0):
            not_touch += 1
        elif (tactile_2 == 0.0):
            not_touch += 1
        elif (tactile_3 == 0.0):
            not_touch += 1
        elif (tactile_3 == 0.0):
            not_touch += 1
        elif (tactile_4 == 0.0):
            not_touch += 1
        elif (tactile_5 == 0.0):
            not_touch += 1
        elif (tactile_6 == 0.0):
            not_touch += 1
        elif (tactile_7 == 0.0):
            not_touch += 1
        elif (tactile_8 == 0.0):
            not_touch += 1
        if (not_touch > 5):
            balance_punishment = 5
        joint_angle_leg_1 = leg1.give_angles() #writing the angles from each leg to a saperate array variable
        joint_angle_leg_2 = leg2.give_angles()
        joint_angle_leg_3 = leg3.give_angles()
        joint_angle_leg_4 = leg4.give_angles()
        joint_angle_leg_5 = leg5.give_angles()
        joint_angle_leg_6 = leg6.give_angles()
        joint_angle_leg_7 = leg7.give_angles()
        joint_angle_leg_8 = leg8.give_angles()
        joint_angles_input = np.concatenate(joint_angle_leg_1, joint_angle_leg_2, joint_angle_leg_3, joint_angle_leg_4, joint_angle_leg_5, joint_angle_leg_6, joint_angle_leg_7, joint_angle_leg_8)
        joint_angle_input = joint_angle_input.reshape(24, 1)#combined all the input angles into a single coloumn vector for a neural network
        model_1.get_neuron_value(model_1.neuron_i, joint_angle_input) #feeding the input angles into the input layer of the neural network
        output_layer1 = np.array(24)
        input_layerhid = np.array(8)
        output_layerhid = np.array(8)
        input_layerhidcont = np.array(8)
        output_layerhidcont = np.array(8)
        input_layerout = np.array(24)
        output_layerout  =np.array(24)
        for i in range(0, 24, 1):
            output_layer1[i] = model_1.input_method(model_1.neuron_i[i, :], weight_in[:, i])#finding the ouput of the input layer
        for j in range(0, 8, 1):
            output_layerh =  output_layer1[0, 3*j:3*j+3]
            output_layerh.append(output_layerhidcont[j])#adding output from context layer from previous iteration as input for hidden layer
            input_layerhid[j] = model_1.input_method(output_layer1, model_1.weights_i[j, :])
        model_1.get_neuron_value(model_1.neuron_h, input_layerhid)# feeding the neuron value of hidden layer
        output_layerhid = model_1.transfer_function(model_1.neuron_h)
        for i in range(0, 8, 1):
            input_layerhidcont[i] = model_1.input_method(output_layerhid[i],model_1.weights_h_c[i])#input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
        model_1.get_neuron_value(model_1.neuron_h_c, input_layerhidcont)#passing the input to the hidden context
        output_layerhidcont = model_1.transfer_frunction(model_1.neuron_h_c)#output of the neurons from context layer
        input_layerout = tf.multiply(output_layerhid, model_1.weights_o[0:8])
        input_layerout.append(tf.multiply(output_layerhid, model_1.weights_o[8:16]))
        input_layerout.append(tf.multiply(output_layerhid, model_1.weights_o[16:24]))#element wise multiplication
        model_1.get_neuron_value(model_1.neuron_o, input_layerout)
        output_layerout1 = model_1.output_function(model_1.neuron_o)#output of the neural network for model 1 that is to be feeded to the robot
        'NN for model_2'
        for i in range(0, 24, 1):
            output_layer1[i] = model_2.input_method(model_2.neuron_i[i, :], model_2.weight_in[:, i])#finding the ouput of the input layer
        for j in range(0, 8, 1):
            output_layerh =  output_layer1[0, 3*j:3*j+3]
            output_layerh.append(output_layerhidcont[j])#adding output from context layer from previous iteration as input for hidden layer
            input_layerhid[j] = model_2.input_method(output_layer1, model_2.weights_i[j, :])
        model_2.get_neuron_value(model_2.neuron_h, input_layerhid)# feeding the neuron value of hidden layer
        output_layerhid = model_2.transfer_function(model_2.neuron_h)
        for i in range(0, 8, 1):
            input_layerhidcont[i] = model_2.input_method(output_layerhid[i],model_2.weights_h_c[i])#input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
        model_2.get_neuron_value(model_2.neuron_h_c, input_layerhidcont)#passing the input to the hidden context
        output_layerhidcont = model_2.transfer_frunction(model_2.neuron_h_c)#output of the neurons from context layer
        input_layerout = tf.multiply(output_layerhid, model_2.weights_o[0:8])
        input_layerout.append(tf.multiply(output_layerhid, model_2.weights_o[8:16]))
        input_layerout.append(tf.multiply(output_layerhid, model_2.weights_o[16:24]))#element wise multiplication
        model_2.get_neuron_value(model_2.neuron_o, input_layerout)
        output_layerout2 = model_2.output_function(model_2.neuron_o)#output of the neural network for model 1 that is to
        'NN for model 3'
        for i in range(0, 24, 1):
            output_layer1[i] = model_3.input_method(model_3.neuron_i[i, :], weight_in[:, i])#finding the ouput of the input layer
        for j in range(0, 8, 1):
            output_layerh =  output_layer1[0, 3*j:3*j+3]
            output_layerh.append(output_layerhidcont[j])#adding output from context layer from previous iteration as input for hidden layer
            input_layerhid[j] = model_3.input_method(output_layer1, model_3.weights_i[j, :])
        model_3.get_neuron_value(model_3.neuron_h, input_layerhid)# feeding the neuron value of hidden layer
        output_layerhid = model_3.transfer_function(model_3.neuron_h)
        for i in range(0, 8, 1):
            input_layerhidcont[i] = model_3.input_method(output_layerhid[i],model_3.weights_h_c[i])#input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
        model_3.get_neuron_value(model_3.neuron_h_c, input_layerhidcont)#passing the input to the hidden context
        output_layerhidcont = model_3.transfer_frunction(model_3.neuron_h_c)#output of the neurons from context layer
        input_layerout = tf.multiply(output_layerhid, model_3.weights_o[0:8])
        input_layerout.append(tf.multiply(output_layerhid, model_3.weights_o[8:16]))
        input_layerout.append(tf.multiply(output_layerhid, model_3.weights_o[16:24]))#element wise multiplication
        model_3.get_neuron_value(model_3.neuron_o, input_layerout)
        output_layerout3 = model_3.output_function(model_3.neuron_o)#output of the neural network for model 1 that is to
        'NN for model 4'
        for i in range(0, 24, 1):
            output_layer1[i] = model_4.input_method(model_4.neuron_i[i, :], model_4.weight_in[:, i])#finding the ouput of the input layer
        for j in range(0, 8, 1):
            output_layerh =  output_layer1[0, 3*j:3*j+3]
            output_layerh.append(output_layerhidcont[j])#adding output from context layer from previous iteration as input for hidden layer
            input_layerhid[j] = model_4.input_method(output_layer1, model_4.weights_i[j, :])
        model_4.get_neuron_value(model_1.neuron_h, input_layerhid)# feeding the neuron value of hidden layer
        output_layerhid = model_4.transfer_function(model_4.neuron_h)
        for i in range(0, 8, 1):
            input_layerhidcont[i] = model_4.input_method(output_layerhid[i],model_4.weights_h_c[i])#input function for hidden context layer, context layr only has one input to each neuron and one weight associated to each neuron
        model_4.get_neuron_value(model_4.neuron_h_c, input_layerhidcont)#passing the input to the hidden context
        output_layerhidcont = model_4.transfer_frunction(model_4.neuron_h_c)#output of the neurons from context layer
        input_layerout = tf.multiply(output_layerhid, model_4.weights_o[0:8])
        input_layerout.append(tf.multiply(output_layerhid, model_4.weights_o[8:16]))
        input_layerout.append(tf.multiply(output_layerhid, model_4.weights_o[16:24]))#element wise multiplication
        model_4.get_neuron_value(model_4.neuron_o, input_layerout)
        output_layerout4 = model_4.output_function(model_4.neuron_o)#output of the neural network for model 1 that is to
        'knowledge transfer algorithm starts here'
        leg1_carollis = leg1.carolis_term(leg1.m_l, leg1.r_1, leg1.l_1, joint_angle_leg_1)
        leg2_carollis = leg2.carolis_term(leg2.m_l, leg2.r_1, leg2.l_1, joint_angle_leg_2)
        leg3_carollis = leg3.carolis_term(leg3.m_l, leg3.r_1, leg3.l_1, joint_angle_leg_3)
        leg4_carollis = leg4.carolis_term(leg4.m_l, leg4.r_1, leg4.l_1, joint_angle_leg_4)
        leg5_carollis = leg5.carolis_term(leg5.m_l, leg5.r_1, leg5.l_1, joint_angle_leg_5)
        leg6_carollis = leg6.carolis_term(leg6.m_l, leg6.r_1, leg6.l_1, joint_angle_leg_6)
        leg7_carollis = leg7.carolis_term(leg7.m_l, leg7.r_1, leg7.l_1, joint_angle_leg_7)
        leg8_carollis = leg8.carolis_term(leg8.m_l, leg8.r_1, leg8.l_1, joint_angle_leg_8)
        carollis_input = np.array([leg1_carollis, leg2_carollis, leg3_carollis, leg4_carollis, leg5_carollis, leg6_carollis, leg7_carollis, leg8_carollis])
        normalization(carollis_input)
        knowledge_transfer.get_neuron(knowledge_transfer.neuron_input, carollis_input)
        'finding the output of the input layer'
        knowledge_input = knowledge_transfer.input_function(knowledge_transfer.neuron_input, knowledge_transfer.weight_in, knowledge_transfer.bias_in)
        'calculating the input for the hidden layer'
        tf.reshape(knowledge_input, [4, 2])
        knowledge_hidden = knowledge_transfer.input_method(knowledge_input, knowledge_transfer.weight_hid, knowledge_transfer.bias_hid)
        'feeding in the input value of hidden neurons'
        knowledge_transfer.get_neuron(knowledge_transfer.neuron_hidden, knowledge_hidden)
        'calculating the output of hidden layer'
        knowledge_hidden_out = knowledge_transfer.activation(knowledge_transfer.neuron_hidden)
        'calculating the input of output layer'
        np.reshape(knowledge_hidden, [2, 2])
        extra = np.array(knowledge_hidden(1, 3), knowledge_hidden(2, 4))
        knowledge_out = tf.concat([knowledge_hidden, extra], axis=0)
        in_out = knowledge_transfer.input_method(knowledge_out, knowledge_transfer.weight_out, knowledge_transfer.bias_out)
        with tf.Session as s:
            s.run(in_out)
        'feeding the input value of output neurons'
        knowledge_transfer.get_neuron(knowledge_transfer.neuron_out, in_out)
        'finding the softmax output of the neurons'
        #softmax_output = np.array(4)
        softmax_output = knowledge_transfer.out_softmax(knowledge_transfer.neuron_out)#this gives the softmax output and stores it in the newly created array
        model = max(softmax_output)
        if (softmax_output[0] == model):
            try:
                joint_modifier(1, 'joint_1_1', output_layerout1[0])
                joint_modifier(1, 'joint_1_2', output_layerout1[1])
                joint_modifier(1, 'joint_1_3', output_layerout1[2])
                joint_modifier(1, 'joint_4_1', output_layerout1[9])
                joint_modifier(1, 'joint_4_2', output_layerout1[10])
                joint_modifier(1, 'joint_4_3', output_layerout1[11])
                joint_modifier(1, 'joint_5_1', output_layerout1[12])
                joint_modifier(1, 'joint_5_2', output_layerout1[13])
                joint_modifier(1, 'joint_5_3', output_layerout1[14])
                joint_modifier(1, 'joint_8_1', output_layerout1[21])
                joint_modifier(1, 'joint_8_2', output_layerout1[22])
                joint_modifier(1, 'joint_8_3', output_layerout1[23])
                joint_modifier(1, 'joint_2_1', output_layerout1[3])
                joint_modifier(1, 'joint_2_2', output_layerout1[4])
                joint_modifier(1, 'joint_2_3', output_layerout1[5])
                joint_modifier(1, 'joint_3_1', output_layerout1[6])
                joint_modifier(1, 'joint_3_2', output_layerout1[7])
                joint_modifier(1, 'joint_3_3', output_layerout1[8])
                joint_modifier(1, 'joint_6_1', output_layerout1[15])
                joint_modifier(1, 'joint_6_2', output_layerout1[16])
                joint_modifier(1, 'joint_6_3', output_layerout1[17])
                joint_modifier(1, 'joint_7_1', output_layerout1[18])
                joint_modifier(1, 'joint_7_2', output_layerout1[19])
                joint_modifier(1, 'joint_7_3', output_layerout1[20])
                x_left_com = Leg_attribute.x_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                y_left_com = Leg_attribute.y_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                x_right_com = Leg_attribute.x_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                y_right_com = Leg_attribute.y_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                x_system_com = Leg_attribute.x_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                y_system_com = Leg_attribute.y_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout1[0],
                                                          output_layerout1[1], output_layerout1[2], output_layerout1[3],
                                                          output_layerout1[4], output_layerout1[5], output_layerout1[6],
                                                          output_layerout1[7], output_layerout1[8], output_layerout1[9],
                                                          output_layerout1[10], output_layerout1[11],
                                                          output_layerout1[12], output_layerout1[13],
                                                          output_layerout1[14], output_layerout1[15],
                                                          output_layerout1[16], output_layerout1[17],
                                                          output_layerout1[18], output_layerout1[19],
                                                          output_layerout1[20], output_layerout1[21],
                                                          output_layerout1[22], output_layerout1[23])
                x_mid_com = Leg_attribute.mid_point_x(x_left_com, x_right_com)
                y_mid_com = Leg_attribute.mid_point_y(y_left_com, y_right_com)
                reward = prev_reward+ Leg_attribute.slope(x_mid_com, y_mid_com, x_system_com, y_system_com)
                np.append(reward_history, [reward], axis=0)
                Learningrate = 0.1
                model_1.NN_learningReward(Learningrate, reward)
                prev_reward = reward+ balance_punishment
            except rospy.ROSInterruptException:
                pass
            rospy.spin()
        elif (softmax_output[1] == model):
            try:
                joint_modifier(1, 'joint_7_1', output_layerout[18])
                joint_modifier(1, 'joint_7_2', output_layerout[19])
                joint_modifier(1, 'joint_7_3', output_layerout[20])
                joint_modifier(1, 'joint_2_1', output_layerout[3])
                joint_modifier(1, 'joint_2_2', output_layerout[4])
                joint_modifier(1, 'joint_2_3', output_layerout[5])
                joint_modifier(1, 'joint_1_1', output_layerout[0])
                joint_modifier(1, 'joint_1_2', output_layerout[1])
                joint_modifier(1, 'joint_1_3', output_layerout[2])
                joint_modifier(1, 'joint_4_1', output_layerout[9])
                joint_modifier(1, 'joint_4_2', output_layerout[10])
                joint_modifier(1, 'joint_4_3', output_layerout[11])
                joint_modifier(1, 'joint_3_1', output_layerout[6])
                joint_modifier(1, 'joint_3_2', output_layerout[7])
                joint_modifier(1, 'joint_3_3', output_layerout[8])
                joint_modifier(1, 'joint_6_1', output_layerout[15])
                joint_modifier(1, 'joint_6_2', output_layerout[16])
                joint_modifier(1, 'joint_6_3', output_layerout[17])
                joint_modifier(1, 'joint_5_1', output_layerout[12])
                joint_modifier(1, 'joint_5_2', output_layerout[13])
                joint_modifier(1, 'joint_5_3', output_layerout[14])
                joint_modifier(1, 'joint_8_1', output_layerout[21])
                joint_modifier(1, 'joint_8_2', output_layerout[22])
                joint_modifier(1, 'joint_8_3', output_layerout[23])
                x_left_com = Leg_attribute.x_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                y_left_com = Leg_attribute.y_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                x_right_com = Leg_attribute.x_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                y_right_com = Leg_attribute.y_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                x_system_com = Leg_attribute.x_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                y_system_com = Leg_attribute.y_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout2[0],
                                                          output_layerout2[1], output_layerout2[2], output_layerout2[3],
                                                          output_layerout2[4], output_layerout2[5], output_layerout2[6],
                                                          output_layerout2[7], output_layerout2[8], output_layerout2[9],
                                                          output_layerout2[10], output_layerout2[11],
                                                          output_layerout2[12], output_layerout2[13],
                                                          output_layerout2[14], output_layerout2[15],
                                                          output_layerout2[16], output_layerout2[17],
                                                          output_layerout2[18], output_layerout2[19],
                                                          output_layerout2[20], output_layerout2[21],
                                                          output_layerout2[22], output_layerout2[23])
                x_mid_com = Leg_attribute.mid_point_x(x_left_com, x_right_com)
                y_mid_com = Leg_attribute.mid_point_y(y_left_com, y_right_com)
                reward = prev_reward+Leg_attribute.slope(x_mid_com, y_mid_com, x_system_com, y_system_com)
                np.append(reward_history, [reward], axis=0)
                Learningrate = 0.1
                model_1.NN_learningReward(Learningrate, reward)
                prev_reward = reward+balance_punishment
            except rospy.ROSInterruptException:
                pass
            rospy.spin()
        elif (softmax_output[2] == model):
            try:
                joint_modifier(1, 'joint_1_1', output_layerout[0])
                joint_modifier(1, 'joint_1_2', output_layerout[1])
                joint_modifier(1, 'joint_1_3', output_layerout[2])
                joint_modifier(1, 'joint_8_1', output_layerout[21])
                joint_modifier(1, 'joint_8_2', output_layerout[22])
                joint_modifier(1, 'joint_8_3', output_layerout[23])
                joint_modifier(1, 'joint_3_1', output_layerout[6])
                joint_modifier(1, 'joint_3_2', output_layerout[7])
                joint_modifier(1, 'joint_3_3', output_layerout[8])
                joint_modifier(1, 'joint_2_1', output_layerout[3])
                joint_modifier(1, 'joint_2_2', output_layerout[4])
                joint_modifier(1, 'joint_2_3', output_layerout[5])
                joint_modifier(1, 'joint_5_1', output_layerout[12])
                joint_modifier(1, 'joint_5_2', output_layerout[13])
                joint_modifier(1, 'joint_5_3', output_layerout[14])
                joint_modifier(1, 'joint_4_1', output_layerout[9])
                joint_modifier(1, 'joint_4_2', output_layerout[10])
                joint_modifier(1, 'joint_4_3', output_layerout[11])
                joint_modifier(1, 'joint_7_1', output_layerout[18])
                joint_modifier(1, 'joint_7_2', output_layerout[19])
                joint_modifier(1, 'joint_7_3', output_layerout[20])
                joint_modifier(1, 'joint_6_1', output_layerout[15])
                joint_modifier(1, 'joint_6_2', output_layerout[16])
                joint_modifier(1, 'joint_6_3', output_layerout[17])
                x_left_com = Leg_attribute.x_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                      output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                      output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                      output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                      output_layerout3[10], output_layerout3[11], output_layerout3[12],
                                                      output_layerout3[13], output_layerout3[14], output_layerout3[15],
                                                      output_layerout3[16], output_layerout3[17], output_layerout3[18],
                                                      output_layerout3[19], output_layerout3[20], output_layerout3[21],
                                                      output_layerout3[22], output_layerout3[23])
                y_left_com = Leg_attribute.y_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                      output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                      output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                      output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                      output_layerout3[10], output_layerout3[11], output_layerout3[12],
                                                      output_layerout3[13], output_layerout3[14], output_layerout3[15],
                                                      output_layerout3[16], output_layerout3[17], output_layerout3[18],
                                                      output_layerout3[19], output_layerout3[20], output_layerout3[21],
                                                      output_layerout3[22], output_layerout3[23])
                x_right_com = Leg_attribute.x_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                        output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                        output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                        output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                        output_layerout3[10], output_layerout3[11],
                                                        output_layerout3[12], output_layerout3[13],
                                                        output_layerout3[14], output_layerout3[15],
                                                        output_layerout3[16], output_layerout3[17],
                                                        output_layerout3[18], output_layerout3[19],
                                                        output_layerout3[20], output_layerout3[21],
                                                        output_layerout3[22], output_layerout3[23])
                y_right_com = Leg_attribute.y_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                        output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                        output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                        output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                        output_layerout3[10], output_layerout3[11],
                                                        output_layerout3[12], output_layerout3[13],
                                                        output_layerout3[14], output_layerout3[15],
                                                        output_layerout3[16], output_layerout3[17],
                                                        output_layerout3[18], output_layerout3[19],
                                                        output_layerout3[20], output_layerout3[21],
                                                        output_layerout3[22], output_layerout3[23])
                x_system_com = Leg_attribute.x_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                          output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                          output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                          output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                          output_layerout3[10], output_layerout3[11],
                                                          output_layerout3[12], output_layerout3[13],
                                                          output_layerout3[14], output_layerout3[15],
                                                          output_layerout3[16], output_layerout3[17],
                                                          output_layerout3[18], output_layerout3[19],
                                                          output_layerout3[20], output_layerout3[21],
                                                          output_layerout3[22], output_layerout3[23])
                y_system_com = Leg_attribute.y_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout3[0],
                                                          output_layerout3[1], output_layerout3[2], output_layerout3[3],
                                                          output_layerout3[4], output_layerout3[5], output_layerout3[6],
                                                          output_layerout3[7], output_layerout3[8], output_layerout3[9],
                                                          output_layerout3[10], output_layerout3[11],
                                                          output_layerout3[12], output_layerout3[13],
                                                          output_layerout3[14], output_layerout3[15],
                                                          output_layerout3[16], output_layerout3[17],
                                                          output_layerout3[18], output_layerout3[19],
                                                          output_layerout3[20], output_layerout3[21],
                                                          output_layerout3[22], output_layerout3[23])
                x_mid_com = Leg_attribute.mid_point_x(x_left_com, x_right_com)
                y_mid_com = Leg_attribute.mid_point_y(y_left_com, y_right_com)
                reward = prev_reward + Leg_attribute.slope(x_mid_com, y_mid_com, x_system_com, y_system_com)
                np.append(reward_history, [reward], axis=0)
                Learningrate = 0.1
                model_1.NN_learningReward(Learningrate, reward)
                prev_reward  =reward+balance_punishment
            except rospy.ROSInterruptException:
                pass
            rospy.spin()
        elif (softmax_output[3] == model):
            try:
                joint_modifier(1, 'joint_1_1', output_layerout[0])
                joint_modifier(1, 'joint_1_2', output_layerout[1])
                joint_modifier(1, 'joint_1_3', output_layerout[2])
                joint_modifier(1, 'joint_3_1', output_layerout[6])
                joint_modifier(1, 'joint_3_2', output_layerout[7])
                joint_modifier(1, 'joint_3_3', output_layerout[8])
                joint_modifier(1, 'joint_5_1', output_layerout[12])
                joint_modifier(1, 'joint_5_2', output_layerout[13])
                joint_modifier(1, 'joint_5_3', output_layerout[14])
                joint_modifier(1, 'joint_7_1', output_layerout[18])
                joint_modifier(1, 'joint_7_2', output_layerout[19])
                joint_modifier(1, 'joint_7_3', output_layerout[20])
                joint_modifier(1, 'joint_2_1', output_layerout[3])
                joint_modifier(1, 'joint_2_2', output_layerout[4])
                joint_modifier(1, 'joint_2_3', output_layerout[5])
                joint_modifier(1, 'joint_4_1', output_layerout[9])
                joint_modifier(1, 'joint_4_2', output_layerout[10])
                joint_modifier(1, 'joint_4_3', output_layerout[11])
                joint_modifier(1, 'joint_6_1', output_layerout[15])
                joint_modifier(1, 'joint_6_2', output_layerout[16])
                joint_modifier(1, 'joint_6_3', output_layerout[17])
                joint_modifier(1, 'joint_8_1', output_layerout[21])
                joint_modifier(1, 'joint_8_2', output_layerout[22])
                joint_modifier(1, 'joint_8_3', output_layerout[23])
                x_left_com = Leg_attribute.x_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                      output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                      output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                      output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                      output_layerout4[10], output_layerout4[11], output_layerout4[12],
                                                      output_layerout4[13], output_layerout4[14], output_layerout4[15],
                                                      output_layerout4[16], output_layerout4[17], output_layerout4[18],
                                                      output_layerout4[19], output_layerout4[20], output_layerout4[21],
                                                      output_layerout4[22], output_layerout4[23])
                y_left_com = Leg_attribute.y_left_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                      output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                      output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                      output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                      output_layerout4[10], output_layerout4[11], output_layerout4[12],
                                                      output_layerout4[13], output_layerout4[14], output_layerout4[15],
                                                      output_layerout4[16], output_layerout4[17], output_layerout4[18],
                                                      output_layerout4[19], output_layerout4[20], output_layerout4[21],
                                                      output_layerout4[22], output_layerout4[23])
                x_right_com = Leg_attribute.x_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                        output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                        output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                        output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                        output_layerout4[10], output_layerout4[11],
                                                        output_layerout4[12], output_layerout4[13],
                                                        output_layerout4[14], output_layerout4[15],
                                                        output_layerout4[16], output_layerout4[17],
                                                        output_layerout4[18], output_layerout4[19],
                                                        output_layerout4[20], output_layerout4[21],
                                                        output_layerout4[22], output_layerout4[23])
                y_right_com = Leg_attribute.y_right_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                        output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                        output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                        output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                        output_layerout4[10], output_layerout4[11],
                                                        output_layerout4[12], output_layerout4[13],
                                                        output_layerout4[14], output_layerout4[15],
                                                        output_layerout4[16], output_layerout4[17],
                                                        output_layerout4[18], output_layerout4[19],
                                                        output_layerout4[20], output_layerout4[21],
                                                        output_layerout4[22], output_layerout4[23])
                x_system_com = Leg_attribute.x_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                          output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                          output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                          output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                          output_layerout4[10], output_layerout4[11],
                                                          output_layerout4[12], output_layerout4[13],
                                                          output_layerout4[14], output_layerout4[15],
                                                          output_layerout4[16], output_layerout4[17],
                                                          output_layerout4[18], output_layerout4[19],
                                                          output_layerout4[20], output_layerout4[21],
                                                          output_layerout4[22], output_layerout4[23])
                y_system_com = Leg_attribute.y_system_com(leg1.m_l, g, leg1.r_1, leg1.l_1, output_layerout4[0],
                                                          output_layerout4[1], output_layerout4[2], output_layerout4[3],
                                                          output_layerout4[4], output_layerout4[5], output_layerout4[6],
                                                          output_layerout4[7], output_layerout4[8], output_layerout4[9],
                                                          output_layerout4[10], output_layerout4[11],
                                                          output_layerout4[12], output_layerout4[13],
                                                          output_layerout4[14], output_layerout4[15],
                                                          output_layerout4[16], output_layerout4[17],
                                                          output_layerout4[18], output_layerout4[19],
                                                          output_layerout4[20], output_layerout4[21],
                                                          output_layerout4[22], output_layerout4[23])
                x_mid_com = Leg_attribute.mid_point_x(x_left_com, x_right_com)
                y_mid_com = Leg_attribute.mid_point_y(y_left_com, y_right_com)
                reward = prev_reward+Leg_attribute.slope(x_mid_com, y_mid_com, x_system_com, y_system_com)
                np.append(reward_history, [reward], axis=0)
                Learningrate = 0.1
                model_1.NN_learningReward(Learningrate, reward)
                prev_reward = reward+balance_punishment
            except rospy.ROSInterruptException:
                pass
            rospy.spin()

















