import numpy as np
import matplotlib.pyplot as plt

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

# open file and read each line
with open('joint_positions.txt') as file:
    list_of_lines = file.readlines()


# create arrays to store x, y, and z data
x = np.zeros(len(list_of_lines))
y = np.zeros(len(list_of_lines))
z = np.zeros(len(list_of_lines))
i = 0

# create arrays to store joint data
joint1 = np.zeros(len(list_of_lines))
joint2 = np.zeros(len(list_of_lines))
joint3 = np.zeros(len(list_of_lines))
joint4 = np.zeros(len(list_of_lines))
joint5 = np.zeros(len(list_of_lines))

# for each line in the text file, get the joint angles, pass through forward kinematics, and store end effector position
for joint_position_str in list_of_lines:
    
    joint_position_str_list = joint_position_str.split()
    joint_angles = np.array([float(joint_position_str_list[0]), float(joint_position_str_list[1]), float(joint_position_str_list[2]), float(joint_position_str_list[3]), float(joint_position_str_list[4])])


    Tq = kinematics.FK_pox(joint_angles, M_matrix, S_list)
    pos = kinematics.get_pose_from_T(Tq)

    # store end effector position
    x[i] = pos[0]
    y[i] = pos[1]
    z[i] = pos[2]

    # store joint angles
    joint1[i] = joint_angles[0]
    joint2[i] = joint_angles[1]
    joint3[i] = joint_angles[2]
    joint4[i] = joint_angles[3]
    joint5[i] = joint_angles[4]

    i += 1

    


# plot the end effector position

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter3D(x, y, z, )

t = np.arange(0, 500, 1)

plt.show()

fig, axs = plt.subplots(3,2)
axs[0,0].plot(t,joint1)
#axs[0,0].set_xlabel('timestep')
axs[0,0].set_ylabel('radians')
axs[0,0].set_title('Joint 1')

axs[0,1].plot(t, joint2)
#axs[0,1].set_xlabel('timestep')
axs[0,1].set_ylabel('radians')
axs[0,1].set_title('Joint 2')

axs[1,0].plot(t, joint3)
#axs[1,0].set_xlabel('timestep')
axs[1,0].set_ylabel('radians')
axs[1,0].set_title('Joint 3')

axs[1,1].plot(t, joint4)
#axs[1,1].set_xlabel('timestep')
axs[1,1].set_ylabel('radians')
axs[1,1].set_title('Joint 4')

axs[2,0].plot(t, joint5)
#axs[2,0].set_xlabel('timestep')
axs[2,0].set_ylabel('radians')
axs[2,0].set_title('Joint 5')

plt.delaxes(axs[2,1])
plt.show()