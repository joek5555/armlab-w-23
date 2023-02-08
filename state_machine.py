"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
from apriltag_ros.msg import AprilTagDetectionArray # jk edit
import cv2
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

            self.calibration_message= "Camera Calibrated with " + str(self.tag_camera_measurements) + " measurements"
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