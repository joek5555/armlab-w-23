import numpy as np
import cv2


def ThresholdImage(hsv_image, thresh):
    """
    Create a binary image using a hsv threshold
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


def DetectBlocks(image, threshold, morphological_constraints, contour_constraints):
    """
    """
    image1 = ThresholdImage(image, threshold)
    image2 = Morphological(image1, morphological_constraints)
    
    contours, boxes = FindContours(image2, contour_constraints)
    return image1, contours, boxes
    
