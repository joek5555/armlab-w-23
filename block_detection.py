import numpy as np
import cv2


def SortDetectedObjectsFunction(detected_object):
    x = detected_object[0][0]
    y = detected_object[0][1]
    return np.sqrt(np.square(x) + np.square(y))

def Depth2Position(depth_image, camera_object):
    """
    Create an array containing the xyz position of each pixel
    3 channel image, (x,y,z) is channel order
    """
    extended_intrinsic_inverse = np.broadcast_to(camera_object.intrinsic_inverse, (921600, 3, 3))
    camera_frame = np.matmul(extended_intrinsic_inverse, camera_object.pixel_grid)
    z = depth_image.transpose(1,0)
    z = np.ravel(z)
    z = z + camera_object.z_offset
    z = z.reshape(-1,1)
    z = np.broadcast_to(z, (921600, 3))
    z = z.reshape(921600, 3, 1)

    camera_frame = np.multiply(z, camera_frame)
    camera_frame = np.concatenate((camera_frame, np.ones((921600,1,1))), axis = 1)
    
    extended_extrinsic_inverse = np.broadcast_to(camera_object.extrinsic_inverse, (921600, 4, 4))
    world_frame = np.matmul(extended_extrinsic_inverse, camera_frame)
    world_frame = np.matmul(camera_object.world_correction_matrix, world_frame)

    position_image = world_frame.transpose(0,2,1)
    position_image = position_image.reshape(1280,720,3)
    position_image = position_image.transpose(1,0,2)

    camera_object.position_image = position_image

    return position_image


def HSVThreshold(hsv_image, thresh):
    """
    Create a binary mask using a hsv threshold
    """
    #mask = cv2.inRange(depth_data, center_val - width_val, center_val + width_val)
    #masked_img = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    if thresh[0,0] > thresh[0,1]:
        thresh_image_low = cv2.inRange(hsv_image, (0, thresh[1,0].item(), thresh[2,0].item()), (thresh[0,1].item(), thresh[1,1].item(), thresh[2,1].item())) 
        thresh_image_high = cv2.inRange(hsv_image, (thresh[0,0].item(), thresh[1,0].item(), thresh[2,0].item()), (180, thresh[1,1].item(), thresh[2,1].item())) 
        thresh_image = cv2.bitwise_or(thresh_image_low, thresh_image_high)
    else:
        thresh_image = cv2.inRange(hsv_image, (thresh[0,0].item(), thresh[1,0].item(), thresh[2,0].item()), (thresh[0,1].item(), thresh[1,1].item(), thresh[2,1].item()))

    return thresh_image

def PositionThreshold(position_image, thresh):
    """
    Create a binary mask using position threshold
    """
    #mask = cv2.inRange(depth_data, center_val - width_val, center_val + width_val)
    #masked_img = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

    thresh_image = cv2.inRange(position_image, (thresh[0,0].item(), thresh[1,0].item(), thresh[2,0].item()), (thresh[0,1].item(), thresh[1,1].item(), thresh[2,1].item()))

    return thresh_image

def morph_shape(val):
    """
    used to convert between number and cv2 morphological shape
    """
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

def Morphological(image, morphological_constraints):
    """
    apply errosion and dilation to an image
    """
    erosion_size = morphological_constraints[0]
    erosion_shape = morphological_constraints[1]
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    eroded_image = cv2.erode(image, element)

    dilatation_size = morphological_constraints[2]
    dilation_shape = morphological_constraints[3]
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    eroded_dilated_image = cv2.dilate(eroded_image, element)
    
    return eroded_dilated_image

def FindBoundingRectangles(image, contour_constraints):
    """
    find the contours and boxes within the image
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    #cv2.drawContours(rgb_image_copy, contours, -1, (0,0,0) , 1)
    rectangles = []
    for single_contour in contours:
        rectangle = cv2.minAreaRect(single_contour)
        if rectangle[1][0] > contour_constraints[0] and rectangle[1][1] > contour_constraints[0]:
            rectangles.append(rectangle)


    return rectangles


def DetectBlocks(rgb_image, depth_image, camera_object):
    """
    """
    detected_objects = []
    distance_threshold_bool = True
    bgr_image_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(bgr_image_cv, cv2.COLOR_BGR2HSV)

    camera_calibrated = camera_object.camera_calibrated
    if camera_calibrated:
        position_image = Depth2Position(depth_image, camera_object)
        xy_position_mask = PositionThreshold(position_image, camera_object.xy_threshold)

    # color is a tuple with (color_str, color_bgr, color_threshold)
    for color in camera_object.colors:
        hsv_mask = HSVThreshold(hsv_image, color[2])
        if camera_calibrated:
            
            mask = cv2.bitwise_and(hsv_mask, xy_position_mask)
        else:
            mask = hsv_mask

        morphological_mask = Morphological(mask, camera_object.morphological_constraints)
        
        rectangles = FindBoundingRectangles(morphological_mask, camera_object.contour_constraints)

        #cv2.drawContours(rgb_image, contours, -1, (0,0,0) , 1)
        for rectangle in rectangles:

            box = cv2.boxPoints(rectangle)
            box = np.int0(box)
            center = np.array([rectangle[0][0], rectangle[0][1]])
            center = center.astype(int)

            block_size_str = ""
            orientation_str = ""

            if camera_calibrated and distance_threshold_bool:
                center = np.array([rectangle[0][0], rectangle[0][1]])
                center = center.astype(int)
                hsv_s_at_center = hsv_image[center[1], center[0], 1]
                hsv_v_at_center = hsv_image[center[1], center[0], 2]
                print(hsv_v_at_center)
                #rectangle_pixel_coord = np.array([[center[0]], [center[1]], [1]])
                #rectangle_pixel_coord = rectangle_pixel_coord.astype(int)
                #rectangle_world_coord = camera_object.pixel2World(rectangle_pixel_coord)
                center_world = np.array([camera_object.position_image[center[1], center[0], 0],
                                                camera_object.position_image[center[1], center[0], 1],
                                                camera_object.position_image[center[1], center[0], 2] ])

                #z_low = center_world[2] - camera_object.rectangle_z_offset
                #z_high = center_world[2] + camera_object.rectangle_z_offset   
                x_low = center_world[0] - camera_object.rectangle_xy_offset
                x_high = center_world[0] + camera_object.rectangle_xy_offset
                y_low = center_world[1] - camera_object.rectangle_xy_offset
                y_high = center_world[1] + camera_object.rectangle_xy_offset
                
                new_position_threshold = np.array([[x_low, x_high], [y_low, y_high], [-5000,5000]], dtype= np.float32)
                #print(new_position_threshold)
                new_position_mask = PositionThreshold(position_image, new_position_threshold)  
                new_hsv_threshold = np.array([[color[2][0,0], color[2][0,1]], [color[2][1,0], color[2][1,1]], [hsv_v_at_center - 4 , color[2][2,1]]], dtype= np.float32)
                new_hsv_mask = HSVThreshold(hsv_image, new_hsv_threshold)

                new_mask = cv2.bitwise_and(new_hsv_mask, new_position_mask) 

#hsv_s_at_center - 10
#color[2][1,0]

                erosion_kernel_size = 1
                erosion_kernel_shape = 0 # 0 is rectangle
                dilation_kernel_size = 3
                dilation_kernel_shape = 0 # 0 is rectangle
                new_morphological_constraints = np.array([erosion_kernel_size, erosion_kernel_shape, dilation_kernel_size, dilation_kernel_shape])
                new_morphological_mask = Morphological(new_mask, new_morphological_constraints)
                
                #rgb_image[:,:,0] = new_morphological_mask * 100
                #rgb_image[:,:,1] = new_morphological_mask * 0
                #rgb_image[:,:,2] = new_morphological_mask * 0

                #print(np.count_nonzero(new_morphological_mask))
        
                #new_mask = new_position_mask
                ###
                #new_mask_image = new_mask * 255
                #new_mask_image = new_mask_image.reshape(720, 1280, 1)
                #new_mask_image = np.broadcast_to(new_mask_image, (720, 1280, 3))
                #rgb_image = new_mask_image
                ###
                new_rectangles = FindBoundingRectangles(new_morphological_mask, camera_object.contour_constraints)
                if len(new_rectangles) < 1:
                    print("Error: no rectangles detected")
                elif len(new_rectangles) > 1: # if there is more than one rectangle, pick rectangle with largest length and width
                    print("Error: more than one rectangle detected")
                    rectangle = new_rectangles[0]
                    for new_rectangle in new_rectangles: 
                        if new_rectangle[1][0] + new_rectangle[1][1] > rectangle[1][0] + rectangle[1][1]:
                            rectangle = new_rectangle
                else:
                    rectangle = new_rectangles[0]

                
                
                box = cv2.boxPoints(rectangle)
                box = np.int0(box)
                center = np.array([rectangle[0][0], rectangle[0][1]])
                center = center.astype(int)

                box1_world = np.array([position_image[box[0,1], box[0,0], 0], position_image[box[0,1], box[0,0], 1], position_image[box[0,1], box[0,0], 2]])
                box2_world = np.array([position_image[box[1,1], box[1,0], 0], position_image[box[1,1], box[1,0], 1], position_image[box[1,1], box[1,0], 2]])
                box3_world = np.array([position_image[box[2,1], box[2,0], 0], position_image[box[2,1], box[2,0], 1], position_image[box[2,1], box[2,0], 2]])
                distance1 = np.sqrt(np.square(box1_world[0] - box2_world[0]) + np.square(box1_world[1] - box2_world[1]) )
                distance2 = np.sqrt(np.square(box2_world[0] - box3_world[0]) + np.square(box2_world[1] - box3_world[1]) )

                theta = rectangle[2]

                if theta < -45:
                    theta = theta + 90

                orientation_str = str(theta)

                theta = theta * np.pi/180
                #print("theta")
                #print(theta)

                if distance1 > distance2:
                    temp = distance1
                    distance1 = distance2
                    distance2 = temp


                if distance1 < 36:
                    block_size_str = "small "
                    #print("small block")
                else:
                    block_size_str = "large "
                    #print("large block")
                side_ratio = distance1/distance2
                #print("side_ratio")
                #print(side_ratio)

                detected_object = [center_world, theta, block_size_str, color[3]]
                detected_objects.append(detected_object)

            cv2.drawContours(rgb_image, [box], 0, color[1] , 3)
            cv2.putText(rgb_image, block_size_str + color[0] + orientation_str, (center[0] - 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1)

        #if len(rectangles) > 0 and camera_object.camera_calibrated:
        #    center = rectangles[0][0]
        #    rectangle_pixel_coord = np.array([[center[0]], [center[1]], [1]])
        #    rectangle_pixel_coord = rectangle_pixel_coord.astype(int)
        #    rectangle_world_coord = camera_object.pixel2World(rectangle_pixel_coord)
            #print(rectangle_world_coord)

    sorted(detected_objects, key= SortDetectedObjectsFunction)

    return rgb_image, detected_objects
    
