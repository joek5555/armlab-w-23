"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
from apriltag_ros.msg import AprilTagDetectionArray # jk edit
import cv2
import kinematics
#from rxarm import (set_positions)

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.calibration_message = "Camera Not Calibrated"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]

        self.taught_waypoints = []
        self.tag_camera_pose = [0,0,0,0]
        self.tag_camera_measurements = 0
        self.picked_block = False

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record_waypoint":
            self.record_waypoint()
            
        if self.next_state == "clear_waypoints":
            self.clear_waypoints()

        if self.next_state == "exectue_taught_path":
            self.exectue_taught_path()
        
        if self.next_state == "record_gripper_open":
            self.record_gripper_open()

        if self.next_state == "record_gripper_close":
            self.record_gripper_close()

        if self.next_state == "pick_sort_task":
            self.pick_sort_task()
        if self.next_state == "pick_stack_task":
            self.pick_stack_task()
        if self.next_state == "line_up_task":
            self.line_up_task()
        if self.next_state == "stack_colors_task":
            self.stack_colors_task()
        if self.next_state == "stack_high_task":
            self.stack_high_task()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"


        num_states  = len(self.waypoints)
        for i in range(num_states):
            self.rxarm.set_positions(waypoints[i])

    def tag_detections_callback(self, data):
        #tag_detections_sub.unregister()
        #self.tag_camera_pose = []
        self.tag_camera_measurements += 1
        
        for i in range(4):
            if len(data.detections[i].id) == 1:
                position = np.array([data.detections[i].pose.pose.pose.position.x, 
                                     data.detections[i].pose.pose.pose.position.y, 
                                     data.detections[i].pose.pose.pose.position.z])
                self.tag_camera_pose[data.detections[i].id[0] - 1] += position
                #self.tag_camera_pose.insert(data.detections[i].id[0] - 1, np.array([data.detections[i].pose.pose.pose.position.x, 
                                                                              #data.detections[i].pose.pose.pose.position.y, 
                                                                              #data.detections[i].pose.pose.pose.position.z]))   
 


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        
        
        # listen to tag_detections topic
        #rospy.init_node('april_tag_listener', anonymous=True)
        global tag_detections_sub
        tag_detections_sub = rospy.Subscriber("tag_detections", AprilTagDetectionArray, self.tag_detections_callback)
        rospy.sleep(5)
        tag_detections_sub.unregister()
        rospy.sleep(0.1)

        if self.tag_camera_measurements == 0:
            print("Calibration failed, no tag poses received")
            self.calibration_message = "Calibration failed, no tag poses received"
        else:
            for i in range(4):
                self.tag_camera_pose[i] /= self.tag_camera_measurements
            
            print("final tag poses")
            print(self.tag_camera_pose)
            

            #CV2.solvepnp
            
            model_points = np.array([(-0.25, -0.025, 0), (0.25, -0.025, 0), (0.25, 0.275,0),(-0.25, 0.275, 0)])
            image_points = np.zeros((4,2))
            for i in range(4):
                #cramera_pos = self.tag_camera_pose[i].transpose()
                camera_pos = self.tag_camera_pose[i].reshape((3,1))
                image_pos = np.dot(self.camera.intrinsic_matrix, camera_pos)/camera_pos[2]
                image_points[i] = (image_pos[0], image_pos[1])
                
            dist_coeffs = np.zeros((4,1))

            (success, rotation_vector, translation_vector) =cv2.solvePnP(model_points,image_points,self.camera.intrinsic_matrix,dist_coeffs,flags = 0) 
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            #print("Success: ")
            #print(success)
            print("Rotation_matrix")
            print(rotation_matrix)
            print("Translation_vector")
            print(translation_vector)
            print(rotation_matrix.shape)
            translation_vector = translation_vector.reshape((3,1))
            print(translation_vector.shape)
            self.camera.extrinsic_matrix = np.append(rotation_matrix, translation_vector, axis = 1)
            
            self.camera.extrinsic_matrix = np.append(self.camera.extrinsic_matrix, np.array([0,0,0,1]).reshape(1,4), axis = 0)
            self.camera.extrinsic_matrix[0,3] = self.camera.extrinsic_matrix[0,3] * 1000
            self.camera.extrinsic_matrix[1,3] = self.camera.extrinsic_matrix[1,3] * 1000
            self.camera.extrinsic_matrix[2,3] = self.camera.extrinsic_matrix[2,3] * 1000

            self.camera.extrinsic_inverse = np.linalg.inv(self.camera.extrinsic_matrix)
            
            print("Extrinsic Matrix")
            print(self.camera.extrinsic_matrix)
            print("inv Extrinsic Matrix")
            print(self.camera.extrinsic_inverse)

            self.calibration_message= "Calibrated with " + str(self.tag_camera_measurements) + " measures"


            # find homography matrix

            edge_points_world = np.array([[450,-125, 0],[-450,-125, 0],[-450, 425, 0], [450, 425, 0]]) # edge of board
            #edge_points_world = np.array([[250,-25, 0],[-250,-25, 0],[-250, 275, 0], [250, 275, 0]]) # apriltags
            u_min = 149
            u_max = 1131
            v_min = 60
            v_max = 660
            edge_points_pixel = np.zeros((4,2))
            for i in range(4):
                world_point = edge_points_world[i,:]
                
                world_point = np.append(world_point, 1)
                camera_point = np.dot(self.camera.extrinsic_matrix, world_point)
                z_camera_coord = camera_point[2]
                pixel_point = np.dot(self.camera.intrinsic_matrix, camera_point[0:3])
                edge_points_pixel[i,0] =  pixel_point[0] / z_camera_coord
                edge_points_pixel[i,1] =  pixel_point[1] / z_camera_coord

            remap_points_pixel = np.array([[u_max,v_max],[u_min,v_max],[u_min, v_min], [u_max, v_min]])

            self.camera.homography_matrix = cv2.findHomography(edge_points_pixel,remap_points_pixel)[0]


            self.camera.camera_calibrated = True

        self.tag_camera_pose = [0,0,0,0]
        self.tag_camera_measurements = 0
        self.status_message = "Calibration - Completed Calibration"
        
        


    def record_waypoint(self):
        self.current_state = "record waypoint"
        
        self.taught_waypoints.append(self.rxarm.get_positions())
        print("Taught waypoint")
        print(self.taught_waypoints[-1])

        self.next_state = "idle"

    def clear_waypoints(self):
        self.current_state = "clear waypoints"
        
        print("Taught Waypoints before")
        print(self.taught_waypoints)
        self.taught_waypoints = []
        print("Taught Waypoints after clear")
        print(self.taught_waypoints)

        self.next_state = "idle"
    
    def record_gripper_open (self):
        self.current_state = "record_gripper_open"
        self.taught_waypoints.append("open_gripper")
        self.rxarm.open_gripper()
        rospy.sleep(2.5)

        self.next_state = "idle"

    def record_gripper_close(self):
        self.current_state = "record_gripper_close"
        
        self.taught_waypoints.append("close_gripper")
        self.rxarm.close_gripper()
        rospy.sleep(2.5)

        #print("Taught waypoint")
        #print(self.taught_waypoints[-1])

        self.next_state = "idle"

    def exectue_taught_path(self):
        self.current_state = "exectue_taught_path"

        joint_positions_file = open('joint_positions.txt', 'w')

    
        num_states  = len(self.taught_waypoints)
        for i in range(num_states):
            if self.taught_waypoints[i] == "open_gripper":
                self.rxarm.open_gripper()
                for j in range(25):
                    position = self.rxarm.get_positions()
                    joint_positions_string = ""
                    for k in range (5):
                        joint_positions_string += str(position[k])
                        joint_positions_string += " "

                    joint_positions_file.write(joint_positions_string)
                    joint_positions_file.write("\n")
                    print(joint_positions_string)
                    print()
                    rospy.sleep(0.1)

            elif self.taught_waypoints[i] == "close_gripper":
                self.rxarm.close_gripper()
                for j in range(25):
                    position = self.rxarm.get_positions()
                    joint_positions_string = ""
                    for k in range (5):
                        joint_positions_string += str(position[k])
                        joint_positions_string += " "

                    joint_positions_file.write(joint_positions_string)
                    joint_positions_file.write("\n")
                    print(joint_positions_string)
                    print()
                    rospy.sleep(0.1)
            else:
                self.rxarm.set_positions(self.taught_waypoints[i])
                for j in range(25):
                    position = self.rxarm.get_positions()
                    joint_positions_string = ""
                    for k in range (5):
                        joint_positions_string += str(position[k])
                        joint_positions_string += " "

                    joint_positions_file.write(joint_positions_string)
                    joint_positions_file.write("\n")
                    print(joint_positions_string)
                    print()
                    rospy.sleep(0.1)

        self.next_state = "idle"

    
    def pick(self, xyz_coord, angle=0, is_big = True):
        #is_big = False

        x = xyz_coord[0,0]
        y = xyz_coord[1,0]
        z = xyz_coord[2,0]
        pose = np.array([x, y, z+20])
        kinematics.go_to(pose, False, self)
        '''
        if self.picked_block:
            valid_place = kinematics.place_block(x, y, z, self, is_big)
            if valid_place:
                print("Block plaaced")
            else:
                print("Failed(place)")
        else:
            valid_pick = kinematics.pick_block(x, y, z, angle, self,is_big)
            if valid_pick:
                print("Block picked")
            else:
                print("Failed(pick)")
        '''

        self.next_state = "idle"

    #######################################################
    ################## Competition tasks ##################
    #######################################################

    # go_to(x, y, z, from_top, self, angle = np.pi/2, sleep_time = 4)
    # pick(x, y, z, angle)
    # pick_block(x, y, z, angle, self, is_big = True)
    # detected block = [(x,y,z), theta, size_str, color_num]
    # place_block(x, y, z, self, is_big, angle = np.pi/2)

    def pick_sort_task(self):

        self.current_state = "pick_sort_task"

        print("Running task pick sort")
        small_block_starting_place = np.array([-350,-100,3])
        large_block_starting_place = np.array([345,-100,3])
        small_block_x_offset = np.array([50, 0, 0])
        large_block_x_offset = np.array([-50, 0, 0])
        small_block_z_offset = np.array([0, 0, 26])
        large_block_z_offset = np.array([0, 0, 39])
        small_block_count = 0
        large_block_count = 0

        #move_time = 3.0
        #accel_time = 1.0
        move_time = 1.2
        accel_time = 2.0
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)


        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(2)

        while len(self.camera.detected_blocks) > 0:
            detected_blocks = self.camera.detected_blocks
            num_blocks = len(detected_blocks)
            detected_block = detected_blocks[0]

            checking_for_stacked = 1
            while checking_for_stacked < num_blocks:
                
                if detected_blocks[checking_for_stacked][4] - detected_block[4] > 80:
                    break
                else:
                    if detected_blocks[checking_for_stacked][0][2] > detected_block[0][2] + 12:
                        detected_block = detected_blocks[checking_for_stacked]
                    checking_for_stacked += 1

            

            # move to detected block, pick
            if detected_block[2] == "large":
                print("large")
                print(large_block_count)
                if large_block_count >= 9:
                    print("error: cannot place more than 9 large blocks")
                    break
                else: 
                    if large_block_count < 6:
                        place_xyz = large_block_starting_place + large_block_x_offset * large_block_count
                    else:
                        place_xyz = large_block_starting_place + large_block_x_offset * (large_block_count - 3) + large_block_z_offset

                    if large_block_count == 0:
                        place_xyz[2] = place_xyz[2] - 40
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = True)
                    # move to large_block_starting_place + large_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                        large_block_count = large_block_count +1
                        self.rxarm.set_positions(initial_position)
                        rospy.sleep(1)

            else:
                print("small")
                print(small_block_count)
                if small_block_count >= 9:
                    print("error: cannot place more than 9 small blocks")
                    break
                else:
                    if small_block_count < 6:
                        place_xyz = small_block_starting_place + small_block_x_offset * small_block_count
                    else:
                        place_xyz = small_block_starting_place + small_block_x_offset * (small_block_count - 3) + small_block_z_offset

                    if small_block_count == 0:
                        place_xyz[2] = place_xyz[2] - 20
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = False)
                    # pick(detected_block[0][0], detected_block[0][1], detected_block[0][2], detected_block[1])
                    # move to large_block_starting_place + large_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                        small_block_count = small_block_count +1
                        self.rxarm.set_positions(initial_position)
                        rospy.sleep(1)
        
        print("pick and sort task complete")
        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        self.next_state = "idle"
                
    ######################################################################

    def pick_stack_task(self):
        print("Running task pick stack")
        small1_block_starting_place = np.array([300,-125, 0])
        small2_block_starting_place = np.array([225,-75,0])
        small3_block_starting_place = np.array([150,-25,0])
        large_block_starting_place = np.array([-400,-75,0])
        
        small_block_z_offset = np.array([0, 0, 25])
        large_block_z_offset = np.array([0, 0, 39])
        small_block_count = 0
        large_block_count = 0
        max_large_tower_height = 9
        max_small_tower_height = 9

        place_sleep_time = 1.0

        #move_time = 3.0
        #accel_time = 1.0
        move_time = 1.5
        accel_time = 2.0
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)

        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(2)

        while len(self.camera.detected_blocks) > 0:
            detected_blocks = self.camera.detected_blocks
            num_blocks = len(detected_blocks)
            detected_block = detected_blocks[0]

            checking_for_stacked = 1
            while checking_for_stacked < num_blocks:
                
                if detected_blocks[checking_for_stacked][4] - detected_block[4] > 80:
                    break
                else:
                    if detected_blocks[checking_for_stacked][0][2] > detected_block[0][2] + 12:
                        detected_block = detected_blocks[checking_for_stacked]
                    checking_for_stacked += 1

            # move to detected block, pick
            if detected_block[2] == "large":
                print("large")
                print(large_block_count)
                
                if large_block_count >= max_large_tower_height:
                    print("error: cannot stack more blocks")
                    break
                else:
                    place_xyz = large_block_starting_place + large_block_z_offset * large_block_count
                   
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = True)
                    # move to large_block_starting_place + large_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.move_joint(3, 0, self.rxarm.get_positions(), self, place_sleep_time)
                        kinematics.move_joint(2, np.pi/6, self.rxarm.get_positions(), self, place_sleep_time)
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                        large_block_count = large_block_count +1

            else:
                print("small")
                print(small_block_count)
                
                if small_block_count >= max_small_tower_height:
                    print("error: cannot stack more blocks")
                    break
                elif small_block_count <=2:
                    place_xyz = small1_block_starting_place + small_block_z_offset * small_block_count
                elif small_block_count <=5:
                    place_xyz = small2_block_starting_place + small_block_z_offset * (small_block_count-3)
                elif small_block_count <=8:
                    place_xyz = small3_block_starting_place + small_block_z_offset * (small_block_count -6)
                    
                # move to detected_block[0] with orientation detected_block[1]
                kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = False)
                # move to large_block_starting_place + large_block_count * large_block_offset
                if self.picked_block:
                    kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False)
                    small_block_count = small_block_count +1


        print("pick and stack task complete")
        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        self.next_state = "idle"

        #####################################################

    def line_up_task(self):

        print("Running task line up")

        x_offset_place = -5

        place_location_large_ROYGBV = [np.array([350+ x_offset_place,-100,0]), np.array([300+ x_offset_place,-100,0]), np.array([250+ x_offset_place,-100,0]), np.array([200+ x_offset_place,-100,0]), np.array([150+ x_offset_place,-100,0]), np.array([100+ x_offset_place,-100,0])]
        place_location_small_ROYGBV = [np.array([-350,-100,-1]), np.array([-300,-100,-5]), np.array([-250,-100,-5]), np.array([-200,-100,-5]), np.array([-150,-100,-5]), np.array([-100,-100,-5])]
        large_blocks_placed_ROYGBV = [0,0,0,0,0,0,0]
        small_blocks_placed_ROYGBV = [0,0,0,0,0,0,0]

        #move_time = 3.0
        #accel_time = 1.0
        move_time = 1.5
        accel_time = 2.0
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)

        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(2)

        while len(self.camera.detected_blocks) > 0:
            detected_blocks = self.camera.detected_blocks
            num_blocks = len(detected_blocks)
            detected_block = detected_blocks[0]

            checking_for_stacked = 1
            while checking_for_stacked < num_blocks:
                
                if detected_blocks[checking_for_stacked][4] - detected_block[4] > 80:
                    break
                else:
                    if detected_blocks[checking_for_stacked][0][2] > detected_block[0][2] + 12:
                        detected_block = detected_blocks[checking_for_stacked]
                    checking_for_stacked += 1

            # move to detected block, pick
            if detected_block[2] == "large":
                if large_blocks_placed_ROYGBV[detected_block[3]]:
                    print("error: already placed a large block of color")
                    print(detected_block[3])
                    break
                else:
                    place_xyz = place_location_large_ROYGBV[detected_block[3]]
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = True)
                    rospy.sleep(2)
                    # move to large_block_starting_place + large_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                        large_blocks_placed_ROYGBV[detected_block[3]] = 1
                    
                
            else:
                if small_blocks_placed_ROYGBV[detected_block[3]]:
                    print("error: already placed a small block of color")
                    print(detected_block[3])
                    break
                else:
                    place_xyz = place_location_small_ROYGBV[detected_block[3]]
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = False)
                    rospy.sleep(2)
                    # move to small_block_starting_place + small_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                        small_blocks_placed_ROYGBV[detected_block[3]] = 1


        print("line up task complete")
        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        self.next_state = "idle"
                

    ###########################################################################

    def stack_colors_task(self):
        print("Running task stack colors")


        small_block_stacking_place = np.array([-400,75,12])
        large_block_stacking_place = np.array([400,75,0])
        small_block_z_offset = np.array([0, 0, 27])
        large_block_z_offset = np.array([0, 0, 39])

        small_block_count = 0
        large_block_count = 0

        x_offset_place = -5

        #move_time = 3.0
        #accel_time = 1.0
        move_time = 1.2
        accel_time = 2.0
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)


        place_location_large_ROYGBV = [np.array([350 + x_offset_place,-100,0]), np.array([300 + x_offset_place,-100,0]), np.array([250+ x_offset_place,-100,0]), np.array([200+ x_offset_place,-100,0]), np.array([150+ x_offset_place,-100,1]), np.array([100+ x_offset_place,-100,1])]
        place_location_small_ROYGBV = [np.array([-350,-100,-1]), np.array([-300,-100,-5]), np.array([-250,-100,-5]), np.array([-200,-100,-5]), np.array([-150,-100,-5]), np.array([-100,-100,-5])]
        large_blocks_placed_ROYGBV = [0,0,0,0,0,0,0]
        small_blocks_placed_ROYGBV = [0,0,0,0,0,0,0]

        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)

        while len(self.camera.detected_blocks) > 0:
            #print(len(self.camera.detected_blocks))
            detected_blocks = self.camera.detected_blocks
            num_blocks = len(detected_blocks)
            detected_block = detected_blocks[0]

            checking_for_stacked = 1
            while checking_for_stacked < num_blocks:
                
                if detected_blocks[checking_for_stacked][4] - detected_block[4] > 80:
                    break
                else:
                    if detected_blocks[checking_for_stacked][0][2] > detected_block[0][2] + 12:
                        detected_block = detected_blocks[checking_for_stacked]
                    checking_for_stacked += 1

            # move to detected block, pick
            if detected_block[2] == "large":
                print("large")
                print(detected_block[3])
                if large_blocks_placed_ROYGBV[detected_block[3]]:
                    print("error: already placed a large block of color")
                    print(detected_block[3])
                    break
                else:
                    place_xyz = place_location_large_ROYGBV[detected_block[3]]
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = True)
                    rospy.sleep(1)
                    # move to large_block_starting_place + large_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                        large_blocks_placed_ROYGBV[detected_block[3]] = 1
                        large_block_count += 1
                    
                
            else:
                print("small")
                print(detected_block[3])
                if small_blocks_placed_ROYGBV[detected_block[3]]:
                    print("error: already placed a small block of color")
                    print(detected_block[3])
                    break
                else:
                    place_xyz = place_location_small_ROYGBV[detected_block[3]]
                    # move to detected_block[0] with orientation detected_block[1]
                    kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = False)
                    rospy.sleep(1)
                    # move to small_block_starting_place + small_block_count * large_block_offset
                    if self.picked_block:
                        kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                        small_blocks_placed_ROYGBV[detected_block[3]] = 1
                        small_block_count += 1

        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        print("stacking colors now")
                
        for num_stacked_large_blocks in range(large_block_count):
            place_xyz = large_block_stacking_place + large_block_z_offset * num_stacked_large_blocks

            if large_blocks_placed_ROYGBV[5]:
                kinematics.pick_block(x = place_location_large_ROYGBV[5][0], y = place_location_large_ROYGBV[5][1], z = place_location_large_ROYGBV[5][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[5] = 0
            elif large_blocks_placed_ROYGBV[4]:
                kinematics.pick_block(x = place_location_large_ROYGBV[4][0], y = place_location_large_ROYGBV[4][1], z = place_location_large_ROYGBV[4][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[4] = 0
            elif large_blocks_placed_ROYGBV[3]:
                kinematics.pick_block(x = place_location_large_ROYGBV[3][0], y = place_location_large_ROYGBV[3][1], z = place_location_large_ROYGBV[3][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[3] = 0
            elif large_blocks_placed_ROYGBV[2]:
                kinematics.pick_block(x = place_location_large_ROYGBV[2][0], y = place_location_large_ROYGBV[2][1], z = place_location_large_ROYGBV[2][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[2] = 0
            elif large_blocks_placed_ROYGBV[1]:
                kinematics.pick_block(x = place_location_large_ROYGBV[1][0], y = place_location_large_ROYGBV[1][1], z = place_location_large_ROYGBV[1][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[1] = 0
            elif large_blocks_placed_ROYGBV[0]:
                kinematics.pick_block(x = place_location_large_ROYGBV[0][0], y = place_location_large_ROYGBV[0][1], z = place_location_large_ROYGBV[0][2]+ 40, angle = np.pi/2, self= self, is_big = True)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_blocks_placed_ROYGBV[0] = 0
            # move to large_block_starting_place + large_block_count * large_block_offset
            #kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)


        for num_stacked_small_blocks in range(small_block_count):
            place_xyz = small_block_stacking_place + small_block_z_offset * num_stacked_small_blocks 
            
            if small_blocks_placed_ROYGBV[5]:
                kinematics.pick_block(x = place_location_small_ROYGBV[5][0] - 15, y = place_location_small_ROYGBV[5][1], z = place_location_small_ROYGBV[5][2] + 25, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[5] = 0
            elif small_blocks_placed_ROYGBV[4]:
                kinematics.pick_block(x = place_location_small_ROYGBV[4][0] - 15, y = place_location_small_ROYGBV[4][1], z = place_location_small_ROYGBV[4][2] + 25, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[4] = 0
            elif small_blocks_placed_ROYGBV[3]:
                kinematics.pick_block(x = place_location_small_ROYGBV[3][0] - 15, y = place_location_small_ROYGBV[3][1], z = place_location_small_ROYGBV[3][2]+ 25, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[3] = 0
            elif small_blocks_placed_ROYGBV[2]:
                kinematics.pick_block(x = place_location_small_ROYGBV[2][0] - 15, y = place_location_small_ROYGBV[2][1], z = place_location_small_ROYGBV[2][2]+ 25, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[2] = 0
            elif small_blocks_placed_ROYGBV[1]:
                kinematics.pick_block(x = place_location_small_ROYGBV[1][0] - 15, y = place_location_small_ROYGBV[1][1], z = place_location_small_ROYGBV[1][2]+ 25, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[1] = 0
            elif small_blocks_placed_ROYGBV[0]:
                kinematics.pick_block(x = place_location_small_ROYGBV[0][0] - 10, y = place_location_small_ROYGBV[0][1], z = place_location_small_ROYGBV[0][2]+ 28, angle = -np.pi/2, self= self, is_big = False)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)
                small_blocks_placed_ROYGBV[0] = 0

            #kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = -np.pi/2, self= self, is_big = False)
            # move to large_block_starting_place + large_block_count * large_block_offset
            #kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = False, angle= -np.pi/2)





        print("stack color task complete")
        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        self.next_state = "idle"
        

    def stack_high_task(self):
        print("Running task stack high")

        #large_block_starting_place = np.array([300,-75,0]) 
        large_block_starting_place = np.array([-400,-75,0]) 
        large_block_z_offset = np.array([0, 0, 40])
        large_block_count = 0


        place_sleep_time = 1.0

        #move_time = 3.0
        #accel_time = 1.0
        move_time = 1.5
        accel_time = 2.0
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)

        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(2)

        while len(self.camera.detected_blocks) > 0:
            detected_blocks = self.camera.detected_blocks
            num_blocks = len(detected_blocks)
            detected_block = detected_blocks[0]

            checking_for_stacked = 1
            while checking_for_stacked < num_blocks:
                
                if detected_blocks[checking_for_stacked][4] - detected_block[4] > 80:
                    break
                else:
                    if detected_blocks[checking_for_stacked][0][2] > detected_block[0][2] + 12:
                        detected_block = detected_blocks[checking_for_stacked]
                    checking_for_stacked += 1


                


            place_xyz = large_block_starting_place + large_block_z_offset * large_block_count
            
            # move to detected_block[0] with orientation detected_block[1]
            kinematics.pick_block(x = detected_block[0][0], y = detected_block[0][1], z = detected_block[0][2], angle = detected_block[1], self= self, is_big = True)
            # move to large_block_starting_place + large_block_count * large_block_offset
            if self.picked_block:
                kinematics.move_joint(3, 0, self.rxarm.get_positions(), self, place_sleep_time)
                kinematics.move_joint(2, np.pi/6, self.rxarm.get_positions(), self, place_sleep_time)
                kinematics.place_block(x=place_xyz[0], y=place_xyz[1], z=place_xyz[2], self=self, is_big = True, angle= np.pi/2)
                large_block_count = large_block_count +1



        print("stack high task complete")
        initial_position = np.array([0.0, -1.3962, -0.7853, -1.5708, 0.0])
        self.rxarm.set_positions(initial_position)
        rospy.sleep(1)
        self.next_state = "idle"




    #######################################################

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str,str)
    #updateCalibrationMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message,self.sm.calibration_message)
            #self.updateCalibrationMessage.emit(self.sm.calibration_message)
            rospy.sleep(0.05)
