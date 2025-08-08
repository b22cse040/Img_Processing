import numpy as np
import cv2
from sharpening_highpass_filtering import highpass_filter
from smoothing_low_pass_filters import smooth_gaussian_filter, smooth_box_filter

def enhance_sec_derivative(img: np.ndarray, filter: int = -8) -> np.ndarray:
    '''
    img = input image of the form np.ndarray
    filter = central value of the filter
    '''
    c = 1.0 if filter > 0 else -1.0
    sharpened_img = highpass_filter(img, filter=-8)
    
    enhanced_img = cv2.add(img, sharpened_img) if c > 0 else cv2.subtract(img, sharpened_img)
    enhanced_img = enhanced_img.astype(np.uint8)
    return enhanced_img

def enhance_by_masking(img: np.ndarray, smoothing_mode: str = "gaussian", m: int = 3, n: int = 3) -> np.ndarray:
    '''
    img: input image of the form np.ndarray
    smoothing_mode: smoothing mechanism, either gaussian ("default") or box
    m, n: dim of kernel
    '''
    if smoothing_mode == "gaussian":
        smooth_img = smooth_gaussian_filter(img, m=m, n=n)
    elif smoothing_mode == "box":
        smoothed_img = smooth_box_filter(img, m=m, n=n)

    unsharped_mask = cv2.subtract(img, smooth_img)
    enhanced_img = cv2.add(img, unsharped_mask)
    enhanced_img = enhanced_img.astype(np.uint8)
    return enhanced_img

if __name__ == "__main__":
    file_path = 'src\\test_images\\test2.jpg'

    img = cv2.imread(file_path)
    if img is None:
        print("File Not Found")
        
    cv2.imshow("Original", img)

    enhanced_img = enhance_by_masking(img, "gaussian")
    cv2.imshow("Enhanced", enhanced_img)
    cv2.waitKey(0)