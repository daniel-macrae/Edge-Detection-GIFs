import cv2
import numpy as np


def CannyLines(frame, lower_bound=0, upper_bound=60, dilation=1):
    # canny edge detection on the grayscaled image
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, lower_bound, upper_bound)
    
    if dilation > 1: # if the line thickness we want is > 1 pixel
        kernel = np.ones((dilation, dilation), np.uint8) #also needs to be tuned
        img_dil = cv2.dilate(canny, kernel, iterations=1)
        return img_dil

    return canny

def MixedCannyLines(frame, ranges=[0,100,200,255], dilations = [1,2,4]):
    # grayscale and blur the images
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
  
    # template to which we keep adding the line images
    mixed_result = np.zeros_like(gray) 
    
    for idx, thickness in enumerate(dilations): # loop through all the line thicknesses we want

        canny = cv2.Canny(blur, ranges[idx], ranges[idx+1])  # 
        kernel = np.ones((thickness, thickness), np.uint8)  # make dilation kernel
        img_dil = cv2.dilate(canny, kernel, iterations=1)
        mixed_result = cv2.bitwise_or(mixed_result, img_dil)  # add the lines to the template image

    # reduce back to to 0-255 unit8 image
    mixed_result = np.clip(mixed_result, 0, 255)
    mixed_result = mixed_result.astype('uint8')
    
    return mixed_result



def blurImages(frame, prev_frame, blur=0.2):
    # convert frames to float values, to prevent numeric overflow with 8bit ints
    frame = np.asarray(frame, dtype=np.float32)
    prev_frame = np.asarray(prev_frame, dtype=np.float32) * blur
    
    blurred = frame + prev_frame

    # clip and convert back to 8bit int (so cv2 doesn't scream at me later)
    blurred = np.clip(blurred, 0, 255)
    blurred = blurred.astype('uint8')
    
    return blurred