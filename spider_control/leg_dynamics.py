#!/usr/bin/python3 
import tensorflow as tf
import numpy as np
from matplotlib import pylab
import pylab as plt
import math
import time

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
    def __init__(self, j_angles, velocity, effort, tactile):
        self.j_angles = j_angles  # vector containing the joint angles
        self.velocity = velocity  # vector containing joint velocities
        self.j_efforts = effort  # vector containing efforts of each joint in the leg
        self.tactile = tactile
        self.m = 0.3
        self.g = 9.8
        self.r = 0.1
        self.l = 0.4
        self.L = 0.6
        self.M = 2.8
    def update_angles(self, angles):
        self.j_angles = angles
    def update_velocity(self, veloc):
        self.velocity = veloc
    def update_tactile(self, tac):
        self.tactile = tac
    def update_effort(self, ef):
        self.j_effort = ef 
        


    def x_left_com(self):
        return (1/(2*(self.M/2+6*self.g)))*(self.M/2*self.l+2*self.g*self.r*(3*(math.cos(self.j_angles[2])+math.cos(self.j_angles[8])+math.cos(self.j_angles[14])+math.cos(self.j_angles[20]))+2*(math.cos(self.j_angles[1])+math.cos(self.j_angles[7])+math.cos(self.j_angles[13])+math.cos(self.j_angles[19]))+math.cos(self.j_angles[0])+math.cos(self.j_angles[6])+math.cos(self.j_angles[12])+math.cos(self.j_angles[18])))

   #a,b,c,d,e,f,j,k,l,m,n,o,p are respectively m_b,m_l,L,x_body_com/y_body_com,theta_1_1,theta_1_2,...theta_8_3

    def y_left_com(self):
        return (1/(2*(self.M/2+6*self.g)))*(self.M/2*self.l+2*self.g*self.r*(3*(math.sin(self.j_angles[2])+3*math.sin(self.j_angles[8])+math.sin(self.j_angles[14])+math.sin(self.j_angles[20]))+2*(math.sin(self.j_angles[1])+math.sin(self.j_angles[7])+math.sin(self.j_angles[13])+math.sin(self.j_angles[19]))+math.sin(self.j_angles[0])+math.sin(self.j_angles[6])+math.sin(self.j_angles[12])+math.sin(self.j_angles[18])))

    def x_right_com(self):
        return (1/(2*(self.M/2+6*self.g)))*(3*self.M/2*self.l+2*self.g*self.r*(3*math.cos(self.j_angles[5])+3*math.cos(self.j_angles[11])+3*math.cos(self.j_angles[17])+3*math.cos(self.j_angles[23])+2*math.cos(self.j_angles[4])+2*math.cos(self.j_angles[10])+2*math.cos(self.j_angles[16])+2*math.cos(self.j_angles[22])+math.cos(self.j_angles[3])+math.cos(self.j_angles[9])+math.cos(self.j_angles[15])+math.cos(self.j_angles[21])))

    def y_right_com(self):
        return (1/(2*(self.M/2+6*self.g)))*(3*self.M/2*self.l+2*self.g*self.r*(3*math.sin(self.j_angles[5])+3*math.sin(self.j_angles[11])+3*math.sin(self.j_angles[17])+3*math.sin(self.j_angles[23])+2*math.sin(self.j_angles[4])+2*math.sin(self.j_angles[10])+2*math.sin(self.j_angles[16])+2*math.sin(self.j_angles[22])+math.sin(self.j_angles[3])+math.sin(self.j_angles[9])+math.sin(self.j_angles[15])+math.sin(self.j_angles[21])))

    def x_system_com(self):
        return  (1/(self.M/2+6*self.g*self.r))*(2*self.M/2*self.l+self.g*self.r*(3*(math.cos(self.j_angles[2])+math.cos(self.j_angles[5])+math.cos(self.j_angles[8])+math.cos(self.j_angles[11])+math.cos(self.j_angles[14])+math.cos(self.j_angles[17])+math.cos(self.j_angles[20])+math.cos(self.j_angles[23]))+2*(math.cos(self.j_angles[1])+math.cos(self.j_angles[4])+math.cos(self.j_angles[7])+math.cos(self.j_angles[10])+math.cos(self.j_angles[13])+math.cos(self.j_angles[16])+math.cos(self.j_angles[19])+math.cos(self.j_angles[22]))+math.cos(self.j_angles[0])+math.cos(self.j_angles[3])+math.cos(self.j_angles[6])+math.cos(self.j_angles[9])+math.cos(self.j_angles[12])+math.cos(self.j_angles[15])+math.cos(self.j_angles[18])+math.cos(self.j_angles[21])))

    def y_system_com(self):
        return  (1/(self.M/2+6*self.g*self.r))*(2*self.M/2*self.l+self.g*self.r*(3*(math.sin(self.j_angles[2])+math.sin(self.j_angles[5])+math.sin(self.j_angles[8])+math.sin(self.j_angles[11])+math.sin(self.j_angles[14])+math.sin(self.j_angles[17])+math.sin(self.j_angles[20])+math.sin(self.j_angles[23]))+2*(math.sin(self.j_angles[1])+math.sin(self.j_angles[4])+math.sin(self.j_angles[7])+math.sin(self.j_angles[10])+math.sin(self.j_angles[13])+math.sin(self.j_angles[16])+math.sin(self.j_angles[19])+math.sin(self.j_angles[22]))+math.sin(self.j_angles[0])+math.sin(self.j_angles[3])+math.sin(self.j_angles[6])+math.sin(self.j_angles[9])+math.sin(self.j_angles[12])+math.sin(self.j_angles[15])+math.sin(self.j_angles[18])+math.sin(self.j_angles[21])))

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
    
    def tactile_run(self):
        score = 0
        total = 0
        for element in range(0, len(self.tactile)):
            if(self.tactile[element]>0.5):
                total +=1
        if(total>3):
            score = total
        else:
            score = 0 
        return score
    
    def carollis_input(self):
        term_1_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[0])+2*self.l_1*math.sin(self.j_angles[0])+self.r_1*math.sin(self.j_angles[0]+self.j_angles[1])+self.r_1*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[0]+self.j_angles[1])+self.l_1*math.sin(self.j_angles[1])+self.r_1*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2])))
        term_1_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[0]+self.j_angles[1])+self.r_1*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[0]+self.j_angles[1])+self.l_1*math.sin(self.j_angles[1])+self.r_1*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))
        term_1_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[0]+self.j_angles[1]+self.j_angles[2]))
        term_2_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[3])+2*self.l_1*math.sin(self.j_angles[3])+self.r_1*math.sin(self.j_angles[3]+self.j_angles[4])+self.r_1*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[3]+self.j_angles[4])+self.l_1*math.sin(self.j_angles[4])+self.r_1*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5])))
        term_2_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[3]+self.j_angles[4])+self.r_1*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[3]+self.j_angles[4])+self.l_1*math.sin(self.j_angles[4])+self.r_1*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))
        term_2_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[3]+self.j_angles[4]+self.j_angles[5]))
        term_3_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[6])+2*self.l_1*math.sin(self.j_angles[6])+self.r_1*math.sin(self.j_angles[6]+self.j_angles[7])+self.r_1*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[6]+self.j_angles[7])+self.l_1*math.sin(self.j_angles[7])+self.r_1*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8])))
        term_3_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[6]+self.j_angles[7])+self.r_1*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[6]+self.j_angles[7])+self.l_1*math.sin(self.j_angles[7])+self.r_1*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))
        term_3_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[6]+self.j_angles[7]+self.j_angles[8]))
        term_4_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[9])+2*self.l_1*math.sin(self.j_angles[9])+self.r_1*math.sin(self.j_angles[9]+self.j_angles[10])+self.r_1*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[9]+self.j_angles[10])+self.l_1*math.sin(self.j_angles[10])+self.r_1*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11])))
        term_4_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[9]+self.j_angles[10])+self.r_1*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[9]+self.j_angles[10])+self.l_1*math.sin(self.j_angles[10])+self.r_1*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))
        term_4_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[9]+self.j_angles[10]+self.j_angles[11]))
        term_5_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[12])+2*self.l_1*math.sin(self.j_angles[12])+self.r_1*math.sin(self.j_angles[12]+self.j_angles[13])+self.r_1*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[12]+self.j_angles[13])+self.l_1*math.sin(self.j_angles[13])+self.r_1*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14])))
        term_5_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[12]+self.j_angles[13])+self.r_1*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[12]+self.j_angles[13])+self.l_1*math.sin(self.j_angles[13])+self.r_1*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))
        term_5_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[12]+self.j_angles[13]+self.j_angles[14]))
        term_6_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[15])+2*self.l_1*math.sin(self.j_angles[15])+self.r_1*math.sin(self.j_angles[15]+self.j_angles[16])+self.r_1*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[15]+self.j_angles[16])+self.l_1*math.sin(self.j_angles[16])+self.r_1*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17])))
        term_6_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[15]+self.j_angles[16])+self.r_1*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[15]+self.j_angles[16])+self.l_1*math.sin(self.j_angles[16])+self.r_1*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))
        term_6_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[15]+self.j_angles[16]+self.j_angles[17]))
        term_7_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[18])+2*self.l_1*math.sin(self.j_angles[18])+self.r_1*math.sin(self.j_angles[18]+self.j_angles[19])+self.r_1*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[18]+self.j_angles[19])+self.l_1*math.sin(self.j_angles[19])+self.r_1*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20])))
        term_7_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[18]+self.j_angles[19])+self.r_1*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[18]+self.j_angles[19])+self.l_1*math.sin(self.j_angles[19])+self.r_1*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))
        term_7_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[18]+self.j_angles[19]+self.j_angles[20]))
        term_8_1 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[21])+2*self.l_1*math.sin(self.j_angles[21])+self.r_1*math.sin(self.j_angles[21]+self.j_angles[22])+self.r_1*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[21]+self.j_angles[22])+self.l_1*math.sin(self.j_angles[22])+self.r_1*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23])))
        term_8_2 = (self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[21]+self.j_angles[22])+self.r_1*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))+self.m_l*9.8*self.velocity*(self.r_1*math.sin(self.j_angles[21]+self.j_angles[22])+self.l_1*math.sin(self.j_angles[22])+self.r_1*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))+self.m_l*9.8*self.velocity*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))
        term_8_3 = (self.m_l*9.8*self.velocity*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23])+self.m_l*9.8*self.velocity*math.sin(self.j_angles[21]+self.j_angles[22]+self.j_angles[23]))
        term1 = term_1_1 +term_1_2 + term_1_3
        term2 = term_2_1+term_2_2+term_2_3
        term3 = term_3_1+term_3_2+term_3_3
        term4 = term_4_1+term_4_2+term_4_3
        term5 = term_5_1+term_5_2+term_5_3
        term6 = term_6_1+term_6_2+term_6_3
        term7 = term_7_1+term_7_2+term_7_3
        term8 = term_8_1+term_8_2+term_8_3
        term = np.array([[term1], [term2], [term3], [term4], [term5], [term6], [term7], [term8]], dtype=np.float32)
        return term
    def leg_run(self):
        x_l_com = self.x_left_com()
        y_l_com = self.y_left_com()
        x_r_com = self.x_right_com()
        y_r_com = self.y_right_com()
        x_s_com =  self.x_system_com()
        y_s_com = self.y_system_com()
        x_m_com = self.mid_point_x(x_l_com, x_r_com)
        y_m_com = self.mid_point_y(y_l_com, y_r_com)
        reward_stability = self.slope(x_m_com, y_m_com, x_s_com, y_s_com)
        reward_tactile = self.tactile_run()
        reward = reward_stability+reward_tactile
        return reward