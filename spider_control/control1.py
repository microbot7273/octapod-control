#!/usr/bin/python3 
#from my_network import Architecture
import my_network as nn
import leg_dynamics as ld
import knowledge_transfer as kt
import tensorflow as tf
import numpy as np
import roslib
roslib.load_manifest('spider_control')
from matplotlib import pyplot
#import pyplot as plt
from geometry_msgs import *
import rospy, yaml, sys
#from osrf_msgs.msg import JointCommands
from sensor_msgs.msg import JointState
#from joint_states_listener.srv import *
#from joint_States_listener.srv import ReturnJointStates
import threading
import time
from std_msgs.msg import Float64
from std_msgs.msg import Header
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

#microbot = GoogleAuth()
#microbot.LocalWebserverAuth()
#drive = GoogleDrive(microbot)

g_joint_states = None
g_positions = None
g_pos1 = None
tactile_output = None


def tactile_callback(msg):
    global tactile_output
    tactile_output = msg.data

def joint_callback(data, args):
    global g_positions
    global g_joint_states
    global g_pos1
    simulation_mode = args[0] #to distinguish between training and testing
    model_number = args[1]#to distiguish between the type of model to train incase of training 
    ou = tf.one_hot([model_number-1], 4)
    with tf.Session() as session:
        out = session.run(ou)
    #out = kt.one_hot_encoding(model_number)
    rospy.loginfo(data.position)#testing
    pub_msg = JointState() # Make a new msg to publish results
    pub_msg.header = Header()
    pub_msg.name = data.name
    pub_msg.velocity = [10] * len(data.name)
    pub_msg.effort = [100] * len(data.name)
    g_positions = data.position
    velocity = 10*len(data.name)
    effort = 10*len(data.name)
    carollis_inp = leg.carollis_input()
    print("carollis input is ")#testing
    print(carollis_inp)
    knowledge_out = knowledge.run(carollis_inp)
    model_num = np.where(knowledge_out == np.amax(knowledge_out))
    reward = leg.leg_run()
    if(model_num == 0):
        new_position = model1.nn_run(g_positions)
        pub_msg.position = new_position
        model1.nn_learn(reward)
    elif(model_num == 1):
        new_position = model2.nn_run(g_positions)
        pub_msg.position = new_position
        model2.nn_learn(reward)
    elif(model_num == 2):
        new_position = model3.nn_run(g_positions)
        pub_msg.position = new_position
        model3.nn_learn(reward)
    elif(model_num == 3):
        new_position = model4.nn_run(g_positions)
        pub_msg.position = new_position
        model4.nn_learn(reward)
    if(simulation_mode == 'train'):
        knowledge.learn(out, knowledge_out)
    leg.update_angles(new_position)
    leg.update_effort(effort)
    leg.update_velocity(velocity)
    leg.update_tactile(tactile_output)
    joint_pub.publish(pub_msg)

if __name__=='__main__':
    model1 = nn.Architecture(3, 24, 1)
    model2 = nn.Architecture(3, 24, 2)
    model3 = nn.Architecture(3, 24, 3)
    model4 = nn.Architecture(3, 24, 4)
    knowledge = kt.MultiClassLogistic(8, 3)
    pos = np.random.uniform(low = 0, high =2.5, size=24)
    vel = 10
    eff = 100
    tact = np.zeros(8)
    leg = ld.Leg_attribute(pos, vel, eff, tact)
    mode=input('please enter whether you want to test or train ?')
    if(mode == 'train'):
        model_number = int(input('please enter the model you wish to train i.e. 1, 2, 3, 4 \n '))
    rospy.init_node('joint_logger_node', anonymous = True)
    rospy.Subscriber("/sr_tactile/touch/ff", Float64, tactile_callback)
    rospy.Subscriber("joint_states", JointState, joint_callback, (mode, model_number))
    joint_pub = rospy.Publisher('target_joint_states', JointState, queue_size = 10)
    rospy.spin()

    

