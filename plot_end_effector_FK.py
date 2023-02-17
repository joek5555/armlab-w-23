import numpy as np

import kinematics


# forward kinematics parameters
M_matrix = np.array([(0, -1, 0, 0),
                    (1, 0, 0, 424.15),
                    (0, 0, 1, 303.91),
                    (0, 0, 0, 1)])
S_list = []

w1 = np.array([0, 0, 1])
v1 = np.array([0, 0, 0])
w2 = np.array([-1, 0, 0])
v2 = np.array([0, -103.91, 0])
w3 = np.array([1, 0, 0])
v3 = np.array([0, 303.91, -50])
w4 = np.array([1, 0, 0])
v4 = np.array([0, 303.91, -250])
w5 = np.array([0, 1, 0])
v5 = np.array([-303.91, 0, 0])
S_list.append(kinematics.to_s_matrix(w1,v1))
S_list.append(kinematics.to_s_matrix(w2,v2))
S_list.append(kinematics.to_s_matrix(w3,v3))
S_list.append(kinematics.to_s_matrix(w4,v4))
S_list.append(kinematics.to_s_matrix(w5,v5))


with open('joint_positions.txt') as file:
    list_of_lines = file.readlines()

for joint_position_str in list_of_lines:
    
    joint_position_str_list = joint_position_str.split()
    joint_positions = float(joint_position_str_list)
    print(joint_positions)

    #Tq = kinematics.FK_pox(joint_angles, M_matrix, S_list)
    #pos = kinematics.get_pose_from_T(Tq)
#angles = kinematics.get_euler_angles_from_T(Tq)

#return [pos[0], pos[1], pos[2], angles[0], angles[1], angles[2]]