"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import cv2
import rospy

import math

pick_offset_big = -35
pick_offset_small = -20
place_offset_big = 5
place_offset_small = 5
pick_approach_offset = 40
place_approach_offset = 60

sleep_time_top = 3
sleep_time_side = 5
sleep_time_base = 3
sleep_time_approach = 2

#data of arm in mm
l1 = 103.91
l2_1 = 200
l2_2 = 50
l2 = np.sqrt(l2_1*l2_1 + l2_2*l2_2)
j2_ex = np.arctan2(l2_2,l2_1)
l3 = 200
l4 = 65
l5 = 66 + 43.15


r2d = 180/np.pi
def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.

    """
    xyz = True
    if xyz:
        sy = math.sqrt(T[0,0]*T[0,0]+T[1,0]*T[1,0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(T[2,1], T[2,2])
            y = math.atan2(-T[2,0],sy)
            z = math.atan2(T[1,0],T[0,0])
        else:
            x = math.atan2(-T[1,2], T[1,1])
            y = math.atan2(-T[2,0], sy)
            z = 0

        return(np.array([x,y,z]))
    else:


        #Rot_matrix = Rotation.from_matrix([T[0:3,0:3]])
        #Rot_matrix = Rotation.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        #Rot_matrix = T[0:3,0:3]
        Rot_vector = cv2.Rodrigues(T[0:3,0:3])
    
        Rot_matrix = Rotation.from_rotvec([Rot_vector[0,0], Rot_vector[1,0], Rot_vector[2,0]])
    

        Rot_results = Rot_matrix.as_euler('zyz', degrees = True)

        return Rot_results
        


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]
    pos = np.array([x, y, z])
    return pos


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    Tq = np.identity(4)
    for i in range(5):
        e = expm(s_lst[i]*joint_angles[i])
        Tq = np.dot(Tq,e)
    Tq = np.dot(Tq,m_mat)
    return Tq    


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    s = np.array([  (0, -w3, w2, v1), 
                    (w3, 0, -w1, v2),
                    (-w2, w1, 0, v3),
                    (0, 0, 0, 0)])

    return s


def R1(th):
    R = np.array([(1, 0, 0), (0, np.cos(th), -np.sin(th)), (0, np.sin(th), np.cos(th))],dtype=object)
    return R

def R2(th):
    R = np.array([(np.cos(th), 0, np.sin(th)), (0, 1, 0), (-np.sin(th),0 , np.cos(th))],dtype=object)
    return R

def R3(th):
    R = np.array([(np.cos(th), -np.sin(th), 0), (np.sin(th), np.cos(th), 0), (0, 0, 1)],dtype=object)
    return R

def R_zyz(p1, p2, p3):
    #find zyz Rotation Matrix
    cp1 = np.cos(p1)
    sp1 = np.sin(p1)
    cp2 = np.cos(p2)
    sp2 = np.sin(p2)
    cp3 = np.cos(p3)
    sp3 = np.sin(p3)
    r11 = cp1*cp2*cp3 - sp1*sp3
    r12 = -cp1*cp2*sp3 - sp1*cp3 
    r13 = cp1*sp2
    r21 = sp1*cp2*cp3 + cp1*sp3
    r22 = -sp1*cp2*sp3 + cp1*cp3
    r23 = sp1*sp2
    r31 = -sp2*cp3
    r32 = sp2*sp3
    r33 = cp2
    return np.array([(r11, r12, r13),(r21, r22, r23),(r31, r32, r33)],dtype=object)

def IK_geometric(pose):
    x = pose[0]
    y = pose[1]
    z = pose[2]
    p1 = pose[3]
    p2 = pose[4]
    p3 = pose[5]
    #find zyz Rotation Matrix
    R = R_zyz(p1, p2, p3)
    #print("R:")
    #print(R)
    #print(R)
    #find xc yc zc 
    
    xc = x - (l4+l5)*R[0,2]
    yc = y - (l4+l5)*R[1,2]

    #find j1 j2 j3 when j1 is in range(-90, 90)deg
    x0 = np.sqrt(xc*xc + yc*yc)
    z = z + 0.08229 * x0 + 8.095
        
    zc = z - (l4+l5)*R[2,2]
    y0 = zc - l1
    '''
    print("xc")
    print(xc)
    print("yc)")
    print(yc)
    print("zc")
    print(zc)

    print("x0")
    print(x0)
    print("y0")
    print(y0)
    print("Cos:")
    print((x0*x0 + y0*y0 - l2*l2 -l3*l3)/(2*l2*l3))
    '''
    #j1
    j1_1 = -np.arctan2(xc,yc)
    #j3
    j3_1_1 = np.arccos([(x0*x0 + y0*y0 - l2*l2 -l3*l3)/(2*l2*l3)])
    j3_1_2 = -j3_1_1
    #j2
    j2_1_1 = np.arctan2(y0,x0) - np.arctan2(l3*np.sin(j3_1_1), l2 + l3*np.cos(j3_1_1))
    j2_1_2 = np.arctan2(y0,x0) - np.arctan2(l3*np.sin(j3_1_2), l2 + l3*np.cos(j3_1_2))
    j2_1_1 = np.pi/2 - j2_ex - j2_1_1
    j2_1_2 = np.pi/2 - j2_ex - j2_1_2
    j3_1_1 += np.pi/2 - j2_ex
    j3_1_2 += np.pi/2 - j2_ex
    #find R0_3
    #R0_1_1
    R0_1_1 = R3(j1_1)
    #R1_2
    R1_2_1_1 = np.dot(R2(-np.pi/2),R3(j2_1_1))
    R1_2_1_2 = np.dot(R2(-np.pi/2),R3(j2_1_2))
    #R2_3
    R2_3_1_1 = np.dot(R2(np.pi), R3(j3_1_1))
    R2_3_1_2 = np.dot(R2(np.pi), R3(j3_1_2))
    #R0_3 
    R0_3_1_1 = np.dot(R0_1_1, np.dot(R1_2_1_1, R2_3_1_1))
    R0_3_1_2 = np.dot(R0_1_1, np.dot(R1_2_1_2, R2_3_1_2))

    #find R3_5
    R3_5_1_1 = np.dot(np.transpose(R0_3_1_1), R)
    R3_5_1_2 = np.dot(np.transpose(R0_3_1_2), R)
    #solve j4 and j5
    cj4_1_1 = R3_5_1_1[1, 2]
    sj4_1_1 = - R3_5_1_1[0, 2]
    cj5_1_1 = - R3_5_1_1[2, 1]
    sj5_1_1 = - R3_5_1_1[2, 0]
    j4_1_1 = np.arctan2(sj4_1_1,cj4_1_1)
    j5_1_1 = np.arctan2(sj5_1_1,cj5_1_1)

    cj4_1_2 = R3_5_1_2[1, 2]
    sj4_1_2 = - R3_5_1_2[0, 2]
    cj5_1_2 = - R3_5_1_2[2, 1]
    sj5_1_2 = - R3_5_1_2[2, 0]
    j4_1_2 = np.arctan2(sj4_1_2,cj4_1_2)
    j5_1_2 = np.arctan2(sj5_1_2,cj5_1_2)

    #find j1 j2 j3 when j1 is over -90deg or 90deg
    #j1
    j1_2 = j1_1 + np.pi
    x0 = -x0
    #j3
    j3_2_1 = np.arccos([(x0*x0 + y0*y0 - l2*l2 -l3*l3)/(2*l2*l3)])
    j3_2_2 = -j3_1_1
    #j2
    j2_2_1 = np.arctan2(y0,x0) - np.arctan2(l3*np.sin(j3_2_1), l2 + l3*np.cos(j3_2_1))
    j2_2_2 = np.arctan2(y0,x0) - np.arctan2(l3*np.sin(j3_2_2), l2 + l3*np.cos(j3_2_2))
    j2_2_1 = np.pi/2 - j2_ex - j2_2_1
    j2_2_2 = np.pi/2 - j2_ex - j2_2_2
    j3_2_1 += np.pi/2 - j2_ex
    j3_2_2 += np.pi/2 - j2_ex
    #find R0_3
    #R0_1_1
    R0_1_2 = R3(j1_2)
    #R1_2
    R1_2_2_1 = np.dot(R2(-np.pi/2),R3(j2_2_1))
    R1_2_2_2 = np.dot(R2(-np.pi/2),R3(j2_2_2))
    #R2_3
    R2_3_2_1 = np.dot(R2(np.pi), R3(j3_2_1))
    R2_3_2_2 = np.dot(R2(np.pi), R3(j3_2_2))
    #R0_3 
    R0_3_2_1 = np.dot(R0_1_2, np.dot(R1_2_2_1, R2_3_2_1))
    R0_3_2_2 = np.dot(R0_1_2, np.dot(R1_2_2_2, R2_3_2_2))
    
    #find R3_5
    R3_5_2_1 = np.dot(np.transpose(R0_3_2_1), R)
    R3_5_2_2 = np.dot(np.transpose(R0_3_2_2), R)
    #solve j4 and j5
    cj4_2_1 = R3_5_2_1[1, 2]
    sj4_2_1 = - R3_5_2_1[0, 2]
    cj5_2_1 = - R3_5_2_1[2, 1]
    sj5_2_1 = - R3_5_2_1[2, 0]
    j4_2_1 = np.arctan2(sj4_2_1,cj4_2_1)
    j5_2_1 = np.arctan2(sj5_2_1,cj5_2_1)

    cj4_2_2 = R3_5_2_2[1, 2]
    sj4_2_2 = - R3_5_2_2[0, 2]
    cj5_2_2 = - R3_5_2_2[2, 1]
    sj5_2_2 = - R3_5_2_2[2, 0]
    j4_2_2 = np.arctan2(sj4_2_2,cj4_2_2)
    j5_2_2 = np.arctan2(sj5_2_2,cj5_2_2)
    
    j1_1 = clamp(j1_1)
    j2_1_1 = clamp(j2_1_1)
    j3_1_1 = clamp(j3_1_1)
    j4_1_1 = clamp(j4_1_1)
    j5_1_1 = clamp(j5_1_1)
    j2_1_2 = clamp(j2_1_2)
    j3_1_2 = clamp(j3_1_2)
    j4_1_2 = clamp(j4_1_2)
    j5_1_2 = clamp(j5_1_2)
    j1_2 = clamp(j1_2)
    j2_2_1 = clamp(j2_2_1)
    j3_2_1 = clamp(j3_2_1)
    j4_2_1 = clamp(j4_2_1)
    j5_2_1 = clamp(j5_2_1)
    j2_2_2 = clamp(j2_2_2)
    j3_2_2 = clamp(j3_2_2)
    j4_2_2 = clamp(j4_2_2)
    j5_2_2 = clamp(j5_2_2)
    if p2 != np.pi:
        if j1_1 > 0.2613:
            j1_1 += 0.02007*j1_1 + 1.639*np.pi/180
        elif j1_1 < -0.2613:
            j1_1 -= 0.02007*(-j1_1) + 1.639*np.pi/180
    '''
    #print out results
    print("First Combination: ")
    print(j1_1, " ", j2_1_1, " ", j3_1_1, " ", j4_1_1, " ", j5_1_1)
    print("Second Combination: ")
    print(j1_1, " ", j2_1_2, " ", j3_1_2, " ", j4_1_2, " ", j5_1_2)
    print("Third Combination: ")
    print(j1_2, " ", j2_2_1, " ", j3_2_1, " ", j4_2_1, " ", j5_2_1)
    print("Fouth Combination: ")
    print(j1_2, " ", j2_2_2, " ", j3_2_2, " ", j4_2_2, " ", j5_2_2)
    '''
    return np.array([(j1_1, j2_1_1, j3_1_1, j4_1_1, j5_1_1), (j1_1, j2_1_2, j3_1_2, j4_1_2, j5_1_2), 
                     (j1_2, j2_2_1, j3_2_1, j4_2_1, j5_2_1), (j1_2, j2_2_2, j3_2_2, j4_2_2, j5_2_2)])
    
#pose = np.array([0, 50+200+174.15, 200+103.91, np.pi/2, np.pi/2, 0])
#IK_geometric(pose)


def choose_joint_combination(joint_combinations):
    valid_rows = np.ones(4) # valid is 1, invalid is 0
    valid_combinations = []

    max_joint_angles = np.array([2*np.pi, 1.7089, 1.7702, 1.8193, 2.8240])
    min_joint_angles = np.array([-2*np.pi,-1.4849, -1.6183, -2.1368, -2.7535])

    for i in range(4):
        for j in range(5):
            if np.isnan(joint_combinations[i,j]):
                valid_rows[i] = 0
                break
            elif joint_combinations[i,j] > max_joint_angles[j] or joint_combinations[i,j] < min_joint_angles[j]:
                valid_rows[i] = 0
                break
        if valid_rows[i]:
            valid_combinations.append(joint_combinations[i,:])
    
    if len(valid_combinations) < 1:
        return np.array([-1000, -1000, -1000, -1000, -1000])
    elif len(valid_combinations) == 1:
        return valid_combinations[0]
    else:
        desired_combination = valid_combinations[0]
        for i in range(1, len(valid_combinations)):
            if valid_combinations[i][2] < desired_combination[2]: # pick the combination with the smalled j3 angle
                desired_combination = valid_combinations[i]
        return desired_combination

def is_valid(desired_joint_combination):
    if desired_joint_combination[1] == -1000:
        return False
    else:
        return True

def can_from_top(x, y, z, offset):
    z += offset
    R = R_zyz(np.pi/2, np.pi,  0)
    xc = x - (l4+l5)*R[0,2]
    yc = y - (l4+l5)*R[1,2]
    x0 = np.sqrt(xc*xc + yc*yc)
    z = z + 0.08229 * x0 + 8.095        
    zc = z - (l4+l5)*R[2,2]
    y0 = zc - l1
    if abs((x0*x0 + y0*y0 - l2*l2 -l3*l3)/(2*l2*l3)) <= 1:
        print("Pick from top")
        return True
    else:
        print("Pick from side")
        return False
'''
def choose_orientation_pick(x, y, z, angle):
    pick_from_top = False
    #print("Desired orientation:")
    if can_pick_from_top(x, y, z):
        pick_from_top = True
        #print(np.pi/2, "  ", np.pi, "  ", angle)
        return [np.pi/2, np.pi, angle]
    else:
        j1 = -np.arctan2(x,y)
        #print(j1+np.pi/2, "  ", np.pi/2, "  ", 0)
        return [j1+np.pi/2, np.pi/2, 0]

def choose_orientation_place(x, y, z, place_from_top, angle = np.pi/2):
    if(place_from_top):
        return [np.pi/2, np.pi, angle]
    else:
        j1 = -np.arctan2(x,y)
        #print(j1+np.pi/2, "  ", np.pi/2, "  ", 0)
        return [j1+np.pi/2, np.pi/2, 0]
'''
def desired_joint_combination(pose):
    joint_combinations = IK_geometric(pose)
    desired_joint_combination = choose_joint_combination(joint_combinations)
    return desired_joint_combination

def move_joint(joint_num, radian, current_pose, self, sleep_time_base = 0):
    current_pose[joint_num] = radian
    self.rxarm.set_positions(current_pose)
    rospy.sleep(sleep_time_base)
    return

def go_to(pose, from_top, self, angle=np.pi/2, sleep_time = 4, sleep_time_base = 0):
    x = pose[0]
    y = pose[1]
    z = pose[2]
    if from_top:
        pose = np.array([x, y, z, np.pi/2, np.pi, angle])
        if is_valid(desired_joint_combination(pose)):
            move_joint(0, desired_joint_combination(pose)[0], self.rxarm.get_positions(), self, sleep_time_base)
            self.rxarm.set_positions(desired_joint_combination(pose))
            rospy.sleep(sleep_time)
            return True
        else:
            return False
    else:
        j1 = -np.arctan2(x,y)
        pose = np.array([x, y, z, j1+np.pi/2, np.pi/2, 0])
        if is_valid(desired_joint_combination(pose)):
            move_joint(0, desired_joint_combination(pose)[0], self.rxarm.get_positions(), self, sleep_time_base)
            self.rxarm.set_positions(desired_joint_combination(pose))
            rospy.sleep(sleep_time)
            return True
        else:
            return False

def pick_block(x, y, z, angle, self, is_big = True):
    #prevent touching the plate
    if z < 10:
        z = 50
    if is_big:
        pick_offset = pick_offset_big
    else:
        pick_offset = pick_offset_small

    pick_from_top = can_from_top(x,y,z, pick_approach_offset)
    if pick_from_top:
        sleep_time = sleep_time_top
    else:
        sleep_time = sleep_time_side

    pose_approach = np.array([x, y, z+pick_approach_offset])
    valid_approach = go_to(pose_approach, pick_from_top, self, angle,sleep_time, sleep_time_base)
    if valid_approach:
        pose_pick = np.array([x, y, z+pick_offset])
        valid_pick = go_to(pose_pick, pick_from_top, self, angle, sleep_time_approach)
        if valid_pick:
            self.rxarm.close_gripper()
            self.picked_block = True
            go_to(pose_approach, pick_from_top, self, angle, sleep_time_approach)
            move_joint(1, -np.pi/4, self.rxarm.get_positions(), self, sleep_time_base)
            move_joint(2, 0, self.rxarm.get_positions(), self, sleep_time_base)
            if pick_from_top == False:
                store_location = [300, -75, 0]
                place_block(store_location[0], store_location[1], store_location[2],pick_from_top, self)
            return True
        else:
            print("Can't pick block")
            return False
    else:
        print("Can't approach block")
        return False

def place_block(x, y, z, self, is_big, angle = np.pi/2):
    place_from_top = can_from_top(x,y,z, place_approach_offset)

    if place_from_top:
        sleep_time = sleep_time_top
    else:
        sleep_time = sleep_time_side
    if is_big:
        place_offset = place_offset_big
    else:
        place_offset = place_offset_small

    pose_approach = np.array([x, y, z+place_approach_offset])
    valid_approach = go_to(pose_approach, place_from_top, self, angle, sleep_time, sleep_time_base)
    if valid_approach:
        pose_place = np.array([x, y, z+place_offset])
        valid_place = go_to(pose_place, place_from_top, self, angle, sleep_time_approach)
        if valid_place:
            self.rxarm.open_gripper()
            self.picked_block = False
            go_to(pose_approach, place_from_top, self, angle, sleep_time_approach)
            move_joint(1, -np.pi/4, self.rxarm.get_positions(), self, sleep_time_base)
            move_joint(2, 0, self.rxarm.get_positions(), self, sleep_time_base)
            return True
        else:
            print("Can't place block")
            return False
    else:
        print("Can't approach block(place)")
        return False
