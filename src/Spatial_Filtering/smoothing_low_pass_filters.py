import numpy as np
import cv2

def smooth_box_filter(img: np.ndarray, m : int = 3, n : int = 3) -> np.ndarray:
    box_filter = np.ones((m, n), dtype = np.float32) / (m * n)
    smoothed_img = cv2.filter2D(img, -1, box_filter)
    smoothed_img = smoothed_img.astype(np.uint8)
    return smoothed_img

def smooth_gaussian_filter(img: np.ndarray, m: int = 3, n: int = 3) -> np.ndarray:
    blurred_img = cv2.GaussianBlur(img, (m, n), sigmaX=0)
    return blurred_img

if __name__ == "__main__":
    file_path = 'src\\test_images\\test1.jpg'

    img = cv2.imread(file_path)
    if img is None:
        print("File Not Found")
        
    cv2.imshow("Original", img)

    smoothed_img = smooth_gaussian_filter(img=img, m=5, n=5)
    cv2.imshow("Smoothed Image", smoothed_img)
    cv2.waitKey(0)