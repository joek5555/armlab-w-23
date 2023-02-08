"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
from block_detection import DetectBlocks


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DetectionFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.camera_calibrated = False
        #self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix = np.array([(896.861, 0, 660.523), (0, 897.203, 381.419), (0, 0, 1)]) # factory
        self.intrinsic_inverse = np.linalg.inv(self.intrinsic_matrix)
        #self.intrinsic_matrix = np.array([(905.8, 0, 668.8), (0, 911.7, 376.8), (0, 0, 1)]) # calibrated
        #self.intrinsic_matrix = np.array([(913.4, 0, 673.0), (0, 917.4, 377.7), (0, 0, 1)]) # calibrated
        self.extrinsic_matrix = np.eye(4)
        self.extrinsic_inverse = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.grid_points = np.concatenate((self.grid_points, np.zeros_like(self.grid_points[0,:,:]).reshape(1,14,19)), axis = 0)
        self.grid_points = np.concatenate((self.grid_points, np.ones_like(self.grid_points[0,:,:]).reshape(1,14,19)), axis = 0)
        
        self.grid_points2 = np.array([np.ravel(self.grid_points[0,:,:]), np.ravel(self.grid_points[1,:,:]), 
                                      np.ravel(self.grid_points[2,:,:]), np.ravel(self.grid_points[3,:,:])])

        self.z_offset = -13  # amount to add to all z measurements before calibration
        self.z_b = 6.75 # z = my + b, where y is y world value
        self.z_m = -0.05

        
        

        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        self.red_thresh = np.array([[167,4], [111,255], [41,255]], dtype= np.float32)
        self.orange_thresh = np.array([[4,14], [120,255], [47,255]], dtype= np.float32)
        self.yellow_thresh = np.array([[21,27], [158, 255], [68, 255]], dtype= np.float32)
        self.green_thresh = np.array([[65, 88], [100,255], [53, 255]], dtype= np.float32)
        #self.blue_thresh = np.array([[100, 109], [151, 255], [52,255]], dtype= np.float32)
        self.blue_thresh = np.array([[90, 125], [50, 255], [35,255]], dtype= np.float32)
        self.purple_thresh = np.array([[110, 157], [44, 255], [22,255]], dtype= np.float32)
        self.erosion_kernel_size = 1
        self.erosion_kernel_shape = 0 # 0 is rectangle
        self.dilation_kernel_size = 1
        self.dilation_kernel_shape = 0 # 0 is rectangle
        self.morphological_constraints = np.array([self.erosion_kernel_size, self.erosion_kernel_shape, self.dilation_kernel_size, self.dilation_kernel_shape])
        self.min_pixels_for_rectangle = 10
        self.contour_constraints = np.array([self.min_pixels_for_rectangle])

    def pixel2World(self, pixel_coord):

        z = self.DepthFrameRaw[pixel_coord[1,0]][pixel_coord[0,0]] + self.z_offset
        camera_coord = np.ones([4,1])
        camera_coord[0:3,:] = np.dot((z),np.dot(self.intrinsic_inverse, pixel_coord))  
        world_coord = np.dot(self.extrinsic_inverse, camera_coord)
        world_coord[2,0] = world_coord[2,0]  + self.z_m * world_coord[1,0] + self.z_b
        return world_coord


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDetectionFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.DetectionFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        self.GridFrame = self.VideoFrame.copy()
        
        if self.camera_calibrated:

            #edge_points_world = np.array([[450,-125, -10],[-450,-125, -10],[-450, 425, 10], [450, 425, 10]])
            edge_points_world = np.array([[250,-25, -10],[-250,-25, -10],[-250, 275, 10], [250, 275, 10]])
            edge_points_pixel = np.zeros((4,2))
            for i in range(4):
                world_point = edge_points_world[i,:]
                
                world_point = np.append(world_point, 1)
                camera_point = np.dot(self.extrinsic_matrix, world_point)
                pixel_point = np.dot(self.intrinsic_matrix ,np.delete(camera_point, -1))
                z = self.DepthFrameRaw[world_point[1]][world_point[0]] + self.z_offset
                edge_points_pixel[i,0] =  pixel_point[0] / z
                edge_points_pixel[i,1] =  pixel_point[1] / z

            remap_points_pixel = np.array([[890,510],[390,510],[390, 210], [890, 210]])

            H = cv2.findHomography(edge_points_pixel,remap_points_pixel)[0]
            #self.GridFrame = cv2.warpPerspective(self.GridFrame, H, (self.GridFrame.shape[1], self.GridFrame.shape[0]))
            #H = cv2.findAffine(edge_points_pixel[],remap_points_pixel)[0]
            

            grid_camera_coord = np.dot(self.extrinsic_matrix, self.grid_points2)
            z_camera_coord = grid_camera_coord[2,:]
            grid_pixel_coord = np.dot(self.intrinsic_matrix, grid_camera_coord[0:3,:])
            #print(grid_pixel_coord.shape)
            for i in range(np.shape(grid_pixel_coord)[1]):
                cv2.circle(self.GridFrame, (int(grid_pixel_coord[0,i]/z_camera_coord[i]), int(grid_pixel_coord[1,i]/z_camera_coord[i])), 3, (255,0,0), -1)

    def BlockDetection(self):
        #self.DetectionFrame = self.VideoFrame.copy()
        rgb_image = self.VideoFrame.copy()
        rgb_image_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(rgb_image_cv, cv2.COLOR_BGR2HSV)
        mask, contours, boxes = DetectBlocks(hsv_image, self.green_thresh, self.morphological_constraints, self.contour_constraints)
        cv2.drawContours(rgb_image, contours, -1, (0,0,0) , 1)
        for box in boxes:
            cv2.drawContours(rgb_image, [box], 0, (0,255,0) , 3)
        #self.DetectionFrame = cv2.bitwise_and(hsv_image, hsv_image, mask = image)
        #mask_image = np.stack((mask, np.zeros_like(mask), np.zeros_like(mask)), axis = 2)
        self.DetectionFrame = rgb_image


        

class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            self.camera.projectGridInRGBImage()
            grid_frame = self.camera.convertQtGridFrame()
            self.camera.BlockDetection()
            detection_frame = self.camera.convertQtDetectionFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(
                    rgb_frame, depth_frame, tag_frame, grid_frame, detection_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Grid window",
                    cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Detection window",
                    cv2.cvtColor(self.camera.DetectionFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
