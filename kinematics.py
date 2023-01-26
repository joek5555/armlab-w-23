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

import math





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


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass