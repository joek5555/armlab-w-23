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
        self.tag_camera_pose = []

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
        tag_detections_sub.unregister()
        self.tag_camera_pose = []
        
        for i in range(4):
            if len(data.detections[i].id) == 1:
                #position = np.array()
                self.tag_camera_pose.insert(data.detections[i].id[0] - 1, np.array([data.detections[i].pose.pose.pose.position.x, 
                                                                              data.detections[i].pose.pose.pose.position.y, 
                                                                              data.detections[i].pose.pose.pose.position.z]))
                #tag_camera_pose.insert(data.detections[i].id[0] - 1, data.detections[i].pose.pose.pose.position.x)
                
 


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
        rospy.sleep(0.1)
        #print(self.tag_camera_pose)
        

        #CV2.solvepnp
        intrinsic_matrix = np.array([(896.861, 0, 660.523), (0, 897.203, 381.419), (0, 0, 1)])
        model_points = np.array([(-0.25, -0.025, 0), (0.25, -0.025, 0), (0.25, 0.275,0),(-0.25, 0.275, 0)])
        image_points = np.zeros((4,2))
        for i in range(4):
            #cramera_pos = self.tag_camera_pose[i].transpose()
            camera_pos = self.tag_camera_pose[i].reshape((3,1))
            image_pos = np.dot(intrinsic_matrix, camera_pos)/camera_pos[2]
            image_points[i] = (image_pos[0], image_pos[1])
            
        dist_coeffs = np.zeros((4,1))

        (success, rotation_vector, translation_vector) =cv2.solvePnP(model_points,image_points,intrinsic_matrix,dist_coeffs,flags = 0) 
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
        extrinsic = np.append(rotation_matrix, translation_vector, axis = 1)
        
        extrinsic = np.append(extrinsic, np.array([0,0,0,1]).reshape(1,4), axis = 0)
        
        print("Extrinsic Matrix")
        print(extrinsic)


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
    updateStatusMessage = pyqtSignal(str)
    
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
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)