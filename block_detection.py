import numpy as np
import cv2


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

def FindContours(image, contour_constraints):
    """
    find the contours and boxes within the image
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    #cv2.drawContours(rgb_image_copy, contours, -1, (0,0,0) , 1)
    boxes = []
    for single_contour in contours:
        rect = cv2.minAreaRect(single_contour)
        if rect[1][0] > contour_constraints[0] and rect[1][1] > contour_constraints[0]:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
            #cv2.drawContours(rgb_image_copy, [box], 0, (0,255,0) , 3)

    return contours, boxes


def DetectBlocks(hsv_image, depth_image, camera_object):
    """
    """
    
    hsv_mask = HSVThreshold(hsv_image, camera_object.blue_thresh)
    if camera_object.camera_calibrated:
        position_image = Depth2Position(depth_image, camera_object)
        xy_position_mask = PositionThreshold(position_image, camera_object.xy_thresh)
        mask = cv2.bitwise_and(hsv_mask, xy_position_mask)
    else:
        mask = hsv_mask

    morphological_mask = Morphological(mask, camera_object.morphological_constraints)
    
    contours, boxes = FindContours(morphological_mask, camera_object.contour_constraints)
    return contours, boxes
    
